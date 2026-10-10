/**
 * robots/robotMovement.js
 * Responsible for robot movement, tweening, and LOCAL EDGE ML decision execution.
 *
 * Mode handling:
 *   BASELINE:
 *     - Ignores all ML decisions.
 *     - Existing movement behavior remains completely unchanged.
 *   OPTIMIZED:
 *     - Reads latest decision from window.robotDecisionState[robotId].
 *     - MOVE:    Continue at configured robot speed.
 *     - SLOW:    Reduce current movement speed to 50% without changing r.speed in UI.
 *     - WAIT:    Pause temporarily, preserving path/position. Resumes on MOVE/SLOW
 *                or after a 3s safety timeout. Records waiting time in metrics.
 *     - REROUTE: Recalculates alternate path using existing A* avoiding conflict,
 *                without teleportation. If no alternate path exists, falls back to WAIT.
 */

import { assessTemporalResolution } from '../coordination/conflictDetection.js';
import { selectAdaptiveIntervention, arbitrateConflictRoles, getRobotPriorityRank } from '../coordination/adaptiveIntervention.js';

const DECISION_MAX_AGE_MS = 3000;
const REROUTE_COOLDOWN_MS = 1500;
const WAIT_SAFETY_TIMEOUT_MS = 3000;

// Initialize global debug container
if (typeof window !== 'undefined') {
  window.robotMovementDecisionState = window.robotMovementDecisionState || {};
}

const VALID_DECISIONS = ['MOVE', 'SLOW', 'WAIT', 'REROUTE'];

/**
 * Retrieve the latest valid decision for a robot.
 * In BASELINE mode or when decision is stale (>3000ms), invalid, or missing, returns MOVE.
 *
 * @param {string} robotId
 * @param {string} runMode
 * @returns {{ decision: "MOVE"|"SLOW"|"WAIT"|"REROUTE", isFresh: boolean, raw: object|null }}
 */
function getEffectiveDecision(robotId, runMode) {
  if (runMode !== 'OPTIMIZED') {
    return { decision: 'MOVE', isFresh: false, raw: null };
  }

  const record = (typeof window !== 'undefined' && window.robotDecisionState)
    ? window.robotDecisionState[robotId]
    : null;

  if (!record || !record.decision) {
    return { decision: 'MOVE', isFresh: false, raw: null };
  }

  // Validate that decision is one of the 4 valid ARES V2 classes
  if (!VALID_DECISIONS.includes(record.decision)) {
    return { decision: 'MOVE', isFresh: false, raw: record };
  }

  const now = Date.now();
  const age = now - (record.timestamp || 0);

  if (age > DECISION_MAX_AGE_MS) {
    return { decision: 'MOVE', isFresh: false, raw: record };
  }

  // Optimized movement applies ML decisions when confidence >= 0.50 (or when confidence is omitted/valid)
  const confidence = typeof record.confidence === 'number' ? record.confidence : 1.0;
  if (confidence < 0.50) {
    return { decision: 'MOVE', isFresh: true, raw: record };
  }

  return { decision: record.decision, isFresh: true, raw: record };
}

/**
 * Create the movement controller.
 *
 * @param {Phaser.Scene} scene
 * @param {Phaser.Tilemaps.Tilemap} map
 * @param {Object.<string, object>} robots
 * @param {Object.<string, Phaser.GameObjects.Sprite>} robotSprites
 * @param {Object.<string, Phaser.Tweens.Tween|null>} activeTweens
 * @param {{
 *   updateStatusUI: Function,
 *   detectFleetConflicts: Function,
 *   onSegmentTravelled?: (robotId: string, distancePx: number) => void,
 *   onRobotCompleted?:   (robotId: string) => void,
 *   onWaitTime?:         (robotId: string, durationSec: number) => void,
 *   findPath?:           Function,
 *   getRunMode?:         () => string,
 *   getConflicts?:       () => Array<object>,
 *   updatePathVisualization?: (robotId: string) => void
 * }} callbacks
 * @returns {{
 *   startRobotMovement: Function,
 *   moveRobotToNextWaypoint: Function,
 *   finishRobotMovement: Function
 * }}
 */
export function createMovementController(scene, map, robots, robotSprites, activeTweens, callbacks) {

  // Per-robot internal coordination tracking
  const localCoordinationState = new Map();

  function getCoordinationState(robotId) {
    if (!localCoordinationState.has(robotId)) {
      localCoordinationState.set(robotId, {
        decision: 'MOVE',
        applied: false,
        waiting: false,
        effectiveSpeed: robots[robotId]?.speed || 100,
        rerouteCount: 0,
        lastAppliedAt: null,
        lastRerouteAt: 0,
        lastReroutedDecisionTimestamp: null,
        waitStartedAt: null,
        waitTimerEvent: null,
        consecutiveWaitCount: 0,
        mlDecision: 'MOVE',
        finalDecision: 'MOVE',
        decisionReason: 'Nominal operation'
      });
    }
    return localCoordinationState.get(robotId);
  }

  function syncDebugState(robotId) {
    if (typeof window === 'undefined') return;
    const state = getCoordinationState(robotId);
    if (!window.robotMovementDecisionState) {
      window.robotMovementDecisionState = {};
    }
    const r = robots[robotId];
    window.robotMovementDecisionState[robotId] = {
      decision: state.decision,
      applied: state.applied,
      waiting: state.waiting,
      effectiveSpeed: state.effectiveSpeed !== undefined ? state.effectiveSpeed : (r ? (r.speed || 100) : 100),
      waitStartTime: state.waitStartedAt || null,
      rerouteCount: state.rerouteCount,
      lastAppliedAt: state.lastAppliedAt,
      mlDecision: state.mlDecision || state.decision,
      finalDecision: state.finalDecision || state.decision,
      decisionReason: state.decisionReason || 'Nominal operation'
    };
  }

  // ── Finish ────────────────────────────────────────────────────────────────────
  const finishRobotMovement = (robotId) => {
    const r = robots[robotId];
    const sprite = robotSprites[robotId];
    if (!r || !sprite) return;

    activeTweens[robotId] = null;

    const coord = getCoordinationState(robotId);
    if (coord.waitTimerEvent) {
      coord.waitTimerEvent.remove();
      coord.waitTimerEvent = null;
    }
    coord.waiting = false;
    coord.waitStartedAt = null;
    coord.consecutiveWaitCount = 0;
    coord.effectiveSpeed = r.speed || 100;
    syncDebugState(robotId);

    if (r.destination) {
      const finalX = r.destination.x;
      const finalY = r.destination.y;
      const finalTileX = r.destination.tileX;
      const finalTileY = r.destination.tileY;

      if (typeof sprite.setPosition === 'function') {
        sprite.setPosition(finalX, finalY);
      } else {
        sprite.x = finalX;
        sprite.y = finalY;
      }
      r.x = finalX;
      r.y = finalY;
      r.start.x = finalX;
      r.start.y = finalY;
      r.start.tileX = finalTileX;
      r.start.tileY = finalTileY;
      r.previousValidPosition = { x: finalX, y: finalY };
    }

    r.status = 'completed';
    r.timePath = null;

    if (typeof callbacks.onRobotCompleted === 'function') {
      callbacks.onRobotCompleted(robotId);
    }

    callbacks.updateStatusUI();
    callbacks.detectFleetConflicts();

    // Immediately wake up any waiting robot whose conflict has cleared
    for (const [otherId, otherCoord] of localCoordinationState.entries()) {
      if (otherId !== robotId && otherCoord.waiting) {
        const otherRobot = robots[otherId];
        if (otherRobot && otherRobot.status === 'moving' && typeof otherCoord.checkWaitResume === 'function') {
          otherCoord.checkWaitResume();
        }
      }
    }
  };

  // ── Helper: Extract conflict tiles that lie on upcoming trajectory ────────────
  const getUpcomingConflictTiles = (robotId, waypointIndex) => {
    const r = robots[robotId];
    if (!r || !r.path || waypointIndex >= r.path.length) return [];
    const upcoming = r.path.slice(waypointIndex);
    const conflicts = typeof callbacks.getConflicts === 'function' ? callbacks.getConflicts() : [];
    const conflictTiles = [];

    for (const c of conflicts) {
      if (!c || (c.robotA !== robotId && c.robotB !== robotId)) continue;
      const otherId = c.robotA === robotId ? c.robotB : c.robotA;
      const otherRobot = robots[otherId];
      if (otherRobot && otherRobot.status === 'completed') continue;
      if (c.type === 'vertex') {
        const tx = c.tileX !== undefined ? c.tileX : c.navX;
        const ty = c.tileY !== undefined ? c.tileY : c.navY;
        if (upcoming.some((wp) => Math.abs(wp.tileX - tx) < 0.6 && Math.abs(wp.tileY - ty) < 0.6)) {
          if (otherRobot) {
            // Check if other robot has already cleared this conflict tile
            const otherWpIdx = otherRobot.waypointIndex || 0;
            const otherUpcoming = (otherRobot.path && otherWpIdx < otherRobot.path.length)
              ? otherRobot.path.slice(otherWpIdx)
              : [];
            const otherHasUpcoming = otherUpcoming.some((wp) => Math.abs(wp.tileX - tx) < 0.6 && Math.abs(wp.tileY - ty) < 0.6);
            if (!otherHasUpcoming) {
              const otherSprite = robotSprites[otherId];
              const ox = otherSprite ? otherSprite.x : (otherRobot.x ?? 0);
              const oy = otherSprite ? otherSprite.y : (otherRobot.y ?? 0);
              const targetWx = (map && typeof map.tileToWorldX === 'function') ? (map.tileToWorldX(tx) + 16) : (tx * 32 + 16);
              const targetWy = (map && typeof map.tileToWorldY === 'function') ? (map.tileToWorldY(ty) + 16) : (ty * 32 + 16);
              if (Math.hypot(ox - targetWx, oy - targetWy) > 48.0) {
                continue; // Conflict zone cleared by other robot
              }
            }
          }
          conflictTiles.push({ tileX: tx, tileY: ty, navX: c.navX, navY: c.navY });
        }
      } else if (c.type === 'edge') {
        const from = c.robotA === robotId ? c.fromA : c.fromB;
        const to = c.robotA === robotId ? c.toA : c.toB;
        if (from && to) {
          const fx = from.tileX !== undefined ? from.tileX : from.navX;
          const fy = from.tileY !== undefined ? from.tileY : from.navY;
          const tox = to.tileX !== undefined ? to.tileX : to.navX;
          const toy = to.tileY !== undefined ? to.tileY : to.navY;
          if (upcoming.some((wp) => (Math.abs(wp.tileX - fx) < 0.6 && Math.abs(wp.tileY - fy) < 0.6) ||
                                    (Math.abs(wp.tileX - tox) < 0.6 && Math.abs(wp.tileY - toy) < 0.6))) {
            if (fx !== undefined && fy !== undefined) conflictTiles.push({ tileX: fx, tileY: fy });
            if (tox !== undefined && toy !== undefined) conflictTiles.push({ tileX: tox, tileY: toy });
          }
        }
      }
    }
    return conflictTiles;
  };

  // ── Helper: Extract most imminent conflict on upcoming trajectory ────────────
  const getMostImminentUpcomingConflict = (robotId, waypointIndex) => {
    const r = robots[robotId];
    if (!r || !r.path || waypointIndex >= r.path.length) return null;
    const upcoming = r.path.slice(waypointIndex);
    const conflicts = typeof callbacks.getConflicts === 'function' ? callbacks.getConflicts() : [];
    if (!conflicts || conflicts.length === 0) return null;

    const relevant = [];
    for (const c of conflicts) {
      if (!c || (c.robotA !== robotId && c.robotB !== robotId)) continue;
      const otherId = c.robotA === robotId ? c.robotB : c.robotA;
      const otherRobot = robots[otherId];
      if (otherRobot && otherRobot.status === 'completed') continue;

      if (c.type === 'vertex') {
        const tx = c.tileX !== undefined ? c.tileX : c.navX;
        const ty = c.tileY !== undefined ? c.tileY : c.navY;
        if (tx !== undefined && ty !== undefined) {
          if (upcoming.some((wp) => Math.abs(wp.tileX - tx) < 0.6 && Math.abs(wp.tileY - ty) < 0.6)) {
            if (otherRobot) {
              // Check if other robot has already cleared this conflict tile
              const otherWpIdx = otherRobot.waypointIndex || 0;
              const otherUpcoming = (otherRobot.path && otherWpIdx < otherRobot.path.length)
                ? otherRobot.path.slice(otherWpIdx)
                : [];
              const otherHasUpcoming = otherUpcoming.some((wp) => Math.abs(wp.tileX - tx) < 0.6 && Math.abs(wp.tileY - ty) < 0.6);
              if (!otherHasUpcoming) {
                const otherSprite = robotSprites[otherId];
                const ox = otherSprite ? otherSprite.x : (otherRobot.x ?? 0);
                const oy = otherSprite ? otherSprite.y : (otherRobot.y ?? 0);
                const targetWx = (map && typeof map.tileToWorldX === 'function') ? (map.tileToWorldX(tx) + 16) : (tx * 32 + 16);
                const targetWy = (map && typeof map.tileToWorldY === 'function') ? (map.tileToWorldY(ty) + 16) : (ty * 32 + 16);
                if (Math.hypot(ox - targetWx, oy - targetWy) > 48.0) {
                  continue; // Conflict zone cleared!
                }
              }
            }
            const time = (c.robotA === robotId ? c.timeA : c.timeB) ?? 0;
            relevant.push({ conflict: c, time });
          }
        }
      } else if (c.type === 'edge') {
        const from = c.robotA === robotId ? c.fromA : c.fromB;
        const to = c.robotA === robotId ? c.toA : c.toB;
        if (from && to) {
          const fx = from.tileX !== undefined ? from.tileX : from.navX;
          const fy = from.tileY !== undefined ? from.tileY : from.navY;
          const tox = to.tileX !== undefined ? to.tileX : to.navX;
          const toy = to.tileY !== undefined ? to.tileY : to.navY;
          if (upcoming.some((wp) => (Math.abs(wp.tileX - fx) < 0.6 && Math.abs(wp.tileY - fy) < 0.6) ||
                                    (Math.abs(wp.tileX - tox) < 0.6 && Math.abs(wp.tileY - toy) < 0.6))) {
            const time = (c.robotA === robotId ? c.timeA : c.timeB) ?? 0;
            relevant.push({ conflict: c, time });
          }
        }
      }
    }

    if (relevant.length === 0) return null;

    relevant.sort((a, b) => a.time - b.time);
    return relevant[0].conflict;
  };

  // ── Helper: Count direction turns in path ────────────────────────────────────
  const countTurns = (path) => {
    if (!path || path.length < 3) return 0;
    let turns = 0;
    for (let i = 2; i < path.length; i++) {
      const dx1 = path[i - 1].tileX - path[i - 2].tileX;
      const dy1 = path[i - 1].tileY - path[i - 2].tileY;
      const dx2 = path[i].tileX - path[i - 1].tileX;
      const dy2 = path[i].tileY - path[i - 1].tileY;
      if (dx1 !== dx2 || dy1 !== dy2) turns++;
    }
    return turns;
  };

  // ── Calculate Alternate Path for REROUTE ─────────────────────────────────────
  const tryCalculateAlternatePath = (robotId, currentTileX, currentTileY, destTileX, destTileY, conflictTiles) => {
    if (typeof callbacks.findPath !== 'function') return null;

    // Filter out start and destination tiles so A* can depart and arrive
    const safeBlocked = (conflictTiles || []).filter(
      (t) =>
        !(Math.abs(t.tileX - currentTileX) < 0.6 && Math.abs(t.tileY - currentTileY) < 0.6) &&
        !(Math.abs(t.tileX - destTileX) < 0.6 && Math.abs(t.tileY - destTileY) < 0.6)
    );

    const candidates = [];
    const seen = new Set();

    const addCandidate = (p) => {
      if (!p || p.length < 2) return;
      const key = p.map((n) => `${n.tileX},${n.tileY}`).join('|');
      if (seen.has(key)) return;
      seen.add(key);
      candidates.push(p);
    };

    // 1. Direct A* avoiding conflict tiles (preferred shortest conflict-free route)
    const directAlt = callbacks.findPath(currentTileX, currentTileY, destTileX, destTileY, safeBlocked);
    addCandidate(directAlt);

    // 2. Offsets / Detours (1-2 steps)
    const candidateDeltas = [
      { dx: 0, dy: -1 }, { dx: 0, dy: 1 }, { dx: -1, dy: 0 }, { dx: 1, dy: 0 },
      { dx: -1, dy: -1 }, { dx: 1, dy: -1 }, { dx: -1, dy: 1 }, { dx: 1, dy: 1 },
      { dx: 0, dy: -2 }, { dx: 0, dy: 2 }, { dx: -2, dy: 0 }, { dx: 2, dy: 0 }
    ];

    for (const delta of candidateDeltas) {
      const candX = currentTileX + delta.dx;
      const candY = currentTileY + delta.dy;

      if (safeBlocked.some((t) => Math.abs(t.tileX - candX) < 0.6 && Math.abs(t.tileY - candY) < 0.6)) {
        continue;
      }

      const pathToCand = callbacks.findPath(currentTileX, currentTileY, candX, candY, safeBlocked);
      if (!pathToCand || pathToCand.length < 2) continue;

      const pathFromCand = callbacks.findPath(candX, candY, destTileX, destTileY, safeBlocked);
      if (!pathFromCand || pathFromCand.length === 0) continue;

      addCandidate(pathToCand.concat(pathFromCand.slice(1)));
    }

    if (candidates.length === 0) return null;

    // Score candidates by user's criteria:
    // 1. Conflict risk first (minimize overlap with other robots)
    // 2. Additional path distance (shorter path preferred)
    // 3. Unnecessary detour turns (fewer turns preferred)
    const scored = candidates.map((p) => {
      let risk = 0;
      for (const [otherId, other] of Object.entries(robots)) {
        if (otherId === robotId || !other || !other.path || other.status === 'completed') continue;
        const otherUpcoming = other.path.slice(1);
        for (const wp of p) {
          if (otherUpcoming.some((owp) => Math.abs(owp.tileX - wp.tileX) < 0.6 && Math.abs(owp.tileY - wp.tileY) < 0.6)) {
            risk++;
          }
        }
      }
      return {
        path: p,
        risk,
        length: p.length,
        turns: countTurns(p)
      };
    });

    scored.sort((a, b) => {
      if (a.risk !== b.risk) return a.risk - b.risk;
      if (a.length !== b.length) return a.length - b.length;
      return a.turns - b.turns;
    });

    return scored[0]?.path || null;
  };

  // ── Physical Spatial Separation Constants ─────────────────────────────────
  // Robot sprite display size is 64x64 px (2.0m x 2.0m).
  // Two robots visually touch when center-to-center distance <= 64px (2.0m).
  const ROBOT_SPRITE_DIAMETER_PX = 64.0;
  const PHYSICAL_STOP_THRESHOLD_PX = 64.0;  // 2.0m: Touch boundary, must hold
  const PHYSICAL_CLEAR_THRESHOLD_PX = 68.0; // 2.125m: Safe clearance to resume
  const PHYSICAL_SLOW_THRESHOLD_PX = 96.0;  // 3.0m: Proactive speed moderation

  /**
   * Check whether continuing toward (targetX, targetY) would cause this robot
   * to penetrate another robot's 64px physical sprite footprint.
   */
  const checkImmediateObstacleAhead = (robotId, curX, curY, targetX, targetY, thresholdPx = PHYSICAL_STOP_THRESHOLD_PX) => {
    const dx = targetX - curX;
    const dy = targetY - curY;
    const distToTarget = Math.hypot(dx, dy);
    if (distToTarget < 1e-4) return null;

    const ux = dx / distToTarget;
    const uy = dy / distToTarget;

    for (const [otherId, other] of Object.entries(robots)) {
      if (!other || otherId === robotId || other.status === 'completed') continue;
      const otherSprite = robotSprites[otherId];
      const ox = otherSprite ? otherSprite.x : (other.x !== undefined ? other.x : other.start?.x);
      const oy = otherSprite ? otherSprite.y : (other.y !== undefined ? other.y : other.start?.y);
      if (ox === undefined || oy === undefined) continue;

      const rx = ox - curX;
      const ry = oy - curY;
      const dist = Math.hypot(rx, ry);

      // Forward projection along heading
      const forwardProj = rx * ux + ry * uy;

      // Obstacle is in front and within physical boundary
      if (forwardProj > 0 && dist <= thresholdPx) {
        return { otherId, dist, forwardProj };
      }
    }
    return null;
  };

  // ── Physical Collision Detection (BASELINE Condition) ─────────────────────
  // In baseline mode, robots lack predictive edge AI coordination.
  // When center-to-center separation shrinks to <= 38px (~1.18m bumper touch),
  // they physically collide, halt permanently, tint red, and trigger collision events.
  const COLLISION_THRESHOLD_PX = 38.0;
  const collidedPairs = new Set();

  const resetCollisionState = () => {
    collidedPairs.clear();
  };

  /**
   * Check for physical collision with any other active robot (distance <= thresholdPx).
   *
   * @param {string} robotId
   * @param {number} curX
   * @param {number} curY
   * @param {number} thresholdPx
   * @returns {{ otherId: string, dist: number } | null}
   */
  const checkPhysicalCollision = (robotId, curX, curY, thresholdPx = COLLISION_THRESHOLD_PX) => {
    for (const [otherId, other] of Object.entries(robots)) {
      if (!other || otherId === robotId || other.status === 'completed') continue;
      const otherSprite = robotSprites[otherId];
      const ox = otherSprite ? otherSprite.x : (other.x !== undefined ? other.x : other.start?.x);
      const oy = otherSprite ? otherSprite.y : (other.y !== undefined ? other.y : other.start?.y);
      if (ox === undefined || oy === undefined) continue;

      const dist = Math.hypot(ox - curX, oy - curY);
      if (dist <= thresholdPx) {
        return { otherId, dist };
      }
    }
    return null;
  };

  /**
   * Play visual collision burst effect on canvas:
   * shockwave expanding rings (danger red & warm amber) + subtle camera micro-shake.
   *
   * @param {number} midX
   * @param {number} midY
   */
  const playCollisionEffect = (midX, midY) => {
    if (!scene || !scene.add || typeof scene.add.graphics !== 'function') return;

    try {
      const gfx = scene.add.graphics();
      gfx.setDepth(200);

      if (scene.tweens && typeof scene.tweens.addCounter === 'function') {
        scene.tweens.addCounter({
          from: 8,
          to: 54,
          duration: 500,
          ease: 'Quad.easeOut',
          onUpdate: (tw) => {
            const r = tw.getValue();
            const a = Math.max(0, 1.0 - (r - 8) / 46);
            gfx.clear();

            // Outer danger red shockwave ring
            gfx.lineStyle(3, 0xef4444, a);
            gfx.strokeCircle(midX, midY, r);

            // Inner warm amber burst ring
            gfx.lineStyle(2, 0xf59e0b, a * 0.85);
            gfx.strokeCircle(midX, midY, Math.max(4, r * 0.6));

            // Center bright spark
            gfx.fillStyle(0xffffff, a * 0.9);
            gfx.fillCircle(midX, midY, Math.max(2, 7 * (1 - (r - 8) / 46)));
          },
          onComplete: () => {
            gfx.destroy();
          }
        });
      } else {
        gfx.lineStyle(3, 0xef4444, 1.0);
        gfx.strokeCircle(midX, midY, 32);
        if (scene.time && typeof scene.time.delayedCall === 'function') {
          scene.time.delayedCall(500, () => gfx.destroy());
        }
      }

      // Subtle camera micro-shake for physical impact tactile feedback
      if (scene.cameras && scene.cameras.main && typeof scene.cameras.main.shake === 'function') {
        scene.cameras.main.shake(180, 0.005);
      }
    } catch (err) {
      console.warn('[CollisionEffect] Error playing collision effect:', err);
    }
  };

  /**
   * Halt both collided robots, freeze positions, mark status as 'collided',
   * apply red alert tint, and trigger collision notification callbacks.
   *
   * @param {string} robotAId
   * @param {string} robotBId
   */
  const handleRobotCollision = (robotAId, robotBId) => {
    const pairKey = [robotAId, robotBId].sort().join(':');
    if (collidedPairs.has(pairKey)) return;
    collidedPairs.add(pairKey);

    const rA = robots[robotAId];
    const rB = robots[robotBId];
    const spriteA = robotSprites[robotAId];
    const spriteB = robotSprites[robotBId];

    // Immediately stop active tweens so robots do not continue crossing each other
    if (activeTweens[robotAId]) {
      try { activeTweens[robotAId].stop(); } catch (_) {}
      activeTweens[robotAId] = null;
    }
    if (activeTweens[robotBId]) {
      try { activeTweens[robotBId].stop(); } catch (_) {}
      activeTweens[robotBId] = null;
    }

    // Freeze coordinates and set collided status with red damage tint
    if (rA) {
      if (spriteA) {
        rA.x = spriteA.x;
        rA.y = spriteA.y;
        rA.start.x = spriteA.x;
        rA.start.y = spriteA.y;
        if (typeof spriteA.setTint === 'function') {
          spriteA.setTint(0xef4444);
        }
      }
      rA.status = 'collided';
      rA.effectiveSpeed = 0;
    }

    if (rB) {
      if (spriteB) {
        rB.x = spriteB.x;
        rB.y = spriteB.y;
        rB.start.x = spriteB.x;
        rB.start.y = spriteB.y;
        if (typeof spriteB.setTint === 'function') {
          spriteB.setTint(0xef4444);
        }
      }
      rB.status = 'collided';
      rB.effectiveSpeed = 0;
    }

    // Update coordination states
    const coordA = getCoordinationState(robotAId);
    if (coordA) {
      if (coordA.waitTimerEvent) {
        coordA.waitTimerEvent.remove();
        coordA.waitTimerEvent = null;
      }
      coordA.waiting = false;
      coordA.decision = 'COLLIDED';
      coordA.finalDecision = 'COLLIDED';
      coordA.decisionReason = `Physical collision with ${robotBId}`;
      coordA.effectiveSpeed = 0;
      syncDebugState(robotAId);
    }

    const coordB = getCoordinationState(robotBId);
    if (coordB) {
      if (coordB.waitTimerEvent) {
        coordB.waitTimerEvent.remove();
        coordB.waitTimerEvent = null;
      }
      coordB.waiting = false;
      coordB.decision = 'COLLIDED';
      coordB.finalDecision = 'COLLIDED';
      coordB.decisionReason = `Physical collision with ${robotAId}`;
      coordB.effectiveSpeed = 0;
      syncDebugState(robotBId);
    }

    const midX = ((spriteA?.x || rA?.x || 0) + (spriteB?.x || rB?.x || 0)) / 2;
    const midY = ((spriteA?.y || rA?.y || 0) + (spriteB?.y || rB?.y || 0)) / 2;
    playCollisionEffect(midX, midY);

    console.warn(`[COLLISION] ${robotAId} and ${robotBId} collided at (${midX.toFixed(1)}, ${midY.toFixed(1)}) and halted.`);

    if (typeof callbacks.onRobotCollision === 'function') {
      callbacks.onRobotCollision(robotAId, robotBId);
    }

    callbacks.updateStatusUI();
    callbacks.detectFleetConflicts();
  };

  // ── Handle WAIT State ────────────────────────────────────────────────────────
  const enterWaitState = (robotId, waypointIndex, runMode) => {
    const r = robots[robotId];
    const coord = getCoordinationState(robotId);
    const sprite = robotSprites[robotId];

    coord.decision = 'WAIT';
    coord.applied = true;
    coord.waiting = true;
    coord.effectiveSpeed = 0;
    r.effectiveSpeed = 0;
    coord.lastAppliedAt = Date.now();
    coord.consecutiveWaitCount = (coord.consecutiveWaitCount || 0) + 1;
    if (!coord.waitStartedAt) {
      coord.waitStartedAt = Date.now();
    }
    syncDebugState(robotId);

    // Clear existing timer if any
    if (coord.waitTimerEvent) {
      coord.waitTimerEvent.remove();
      coord.waitTimerEvent = null;
    }

    // Periodic check to resume on fresh MOVE/SLOW, timeout, or conflict cleared
    const checkWaitResume = () => {
      if (!r || r.status !== 'moving') {
        if (coord.waitTimerEvent) coord.waitTimerEvent.remove();
        coord.waitTimerEvent = null;
        coord.checkWaitResume = null;
        coord.waiting = false;
        coord.effectiveSpeed = r ? (r.speed || 100) : 100;
        syncDebugState(robotId);
        return;
      }

      const waitElapsedMs = Date.now() - (coord.waitStartedAt || Date.now());
      const currentCheck = getEffectiveDecision(robotId, runMode);

      const relevantConflict = getMostImminentUpcomingConflict(robotId, waypointIndex);
      let conflictCleared = !relevantConflict;

      let otherRobotId = null;
      let otherRobot = null;
      let otherCoord = null;
      let isMutualWait = false;
      let assessment = null;

      if (relevantConflict) {
        otherRobotId = relevantConflict.robotA === robotId ? relevantConflict.robotB : relevantConflict.robotA;
        otherRobot = robots[otherRobotId];
        if (!otherRobot || otherRobot.status === 'completed' || otherRobot.status === 'idle') {
          conflictCleared = true;
        } else {
          otherCoord = getCoordinationState(otherRobotId);
          isMutualWait = Boolean(otherCoord && otherCoord.waiting);
          assessment = relevantConflict.temporalAssessment ||
            (typeof assessTemporalResolution === 'function' ? assessTemporalResolution(relevantConflict, robots[relevantConflict.robotA], robots[relevantConflict.robotB]) : null);
        }
      }

      // Re-evaluate current adaptive intervention against live runtime state
      let reEvaluatedDecision = 'MOVE';
      if (runMode === 'OPTIMIZED' && !conflictCleared && relevantConflict) {
        const reEval = selectAdaptiveIntervention({
          mlDecision: currentCheck.decision,
          conflict: relevantConflict,
          temporalAssessment: assessment,
          robotState: r,
          otherRobotState: otherRobot,
          isOtherWaiting: isMutualWait,
          consecutiveWaitCount: coord.consecutiveWaitCount || 0
        });
        reEvaluatedDecision = reEval.appliedDecision;
      }

      // Mutual wait / Deadlock breaking
      if (isMutualWait && relevantConflict) {
        const timingPossible = Boolean(assessment && assessment.timingResolutionPossible);

        if (!timingPossible) {
          // Spatial conflict cannot be resolved temporally; prefer REROUTE when valid alternate path exists
          const curTileX = (map && typeof map.worldToTileX === 'function')
            ? map.worldToTileX(sprite.x)
            : Math.floor(sprite.x / 32);
          const curTileY = (map && typeof map.worldToTileY === 'function')
            ? map.worldToTileY(sprite.y)
            : Math.floor(sprite.y / 32);
          const upcomingConflictTiles = getUpcomingConflictTiles(robotId, waypointIndex);
          const alternatePath = r.destination
            ? tryCalculateAlternatePath(robotId, curTileX, curTileY, r.destination.tileX, r.destination.tileY, upcomingConflictTiles)
            : null;

          if (alternatePath && alternatePath.length > 1) {
            if (coord.waitTimerEvent) {
              coord.waitTimerEvent.remove();
              coord.waitTimerEvent = null;
            }
            const waitedDurationSec = parseFloat((waitElapsedMs / 1000).toFixed(3));
            if (typeof callbacks.onWaitTime === 'function' && waitedDurationSec > 0) {
              callbacks.onWaitTime(robotId, waitedDurationSec);
            }
            r.path = alternatePath;
            coord.rerouteCount++;
            coord.lastRerouteAt = Date.now();
            coord.decision = 'REROUTE';
            coord.finalDecision = 'REROUTE';
            coord.decisionReason = 'Mutual wait in unresolvable conflict; spatial rerouting applied';
            coord.waiting = false;
            coord.waitStartedAt = null;
            coord.consecutiveWaitCount = 0;
            coord.applied = true;
            syncDebugState(robotId);
            if (typeof callbacks.updatePathVisualization === 'function') {
              callbacks.updatePathVisualization(robotId);
            }
            callbacks.detectFleetConflicts();
            moveRobotToNextWaypoint(robotId, 1);
            return;
          }
        } else {
          // Priority-aware tie-breaking: designated progressing robot releases
          const { isSelfProgressing } = arbitrateConflictRoles(r, otherRobot, relevantConflict);
          if (isSelfProgressing) {
            reEvaluatedDecision = 'MOVE';
          }
        }
      }

      // Check physical clearance directly ahead (skip completed robots)
      let obstacleAhead = null;
      if (runMode === 'OPTIMIZED') {
        const targetTile = r.path && waypointIndex < r.path.length ? r.path[waypointIndex] : null;
        if (targetTile) {
          const targetWorldX = targetTile.worldX !== undefined ? targetTile.worldX : (map.tileToWorldX(targetTile.tileX) + 16);
          const targetWorldY = targetTile.worldY !== undefined ? targetTile.worldY : (map.tileToWorldY(targetTile.tileY) + 16);
          obstacleAhead = checkImmediateObstacleAhead(
            robotId,
            sprite.x,
            sprite.y,
            targetWorldX,
            targetWorldY,
            PHYSICAL_CLEAR_THRESHOLD_PX
          );
        }
      }

      // If space directly ahead is physically blocked by an active robot within 68px, continue holding safely
      if (obstacleAhead) {
        return;
      }

      // Explicit WAIT release conditions:
      // 1. Conflict zone cleared
      // 2. Adaptive re-evaluation permits MOVE or SLOW
      // 3. Fresh external MOVE/SLOW decision
      // 4. Repeated wait cap or safety timeout expired
      const shouldResume =
        conflictCleared ||
        reEvaluatedDecision === 'MOVE' ||
        reEvaluatedDecision === 'SLOW' ||
        (currentCheck.isFresh && (currentCheck.decision === 'MOVE' || currentCheck.decision === 'SLOW')) ||
        coord.consecutiveWaitCount >= 2 ||
        waitElapsedMs >= WAIT_SAFETY_TIMEOUT_MS;

      if (shouldResume) {
        if (coord.waitTimerEvent) {
          coord.waitTimerEvent.remove();
          coord.waitTimerEvent = null;
        }
        coord.checkWaitResume = null;

        const waitedDurationSec = parseFloat((waitElapsedMs / 1000).toFixed(3));
        if (typeof callbacks.onWaitTime === 'function' && waitedDurationSec > 0) {
          callbacks.onWaitTime(robotId, waitedDurationSec);
        }

        const nextSpeed = (reEvaluatedDecision === 'SLOW') ? Math.max(10, (r.speed || 100) * 0.5) : (r.speed || 100);
        coord.waiting = false;
        coord.waitStartedAt = null;
        coord.consecutiveWaitCount = 0;
        coord.effectiveSpeed = nextSpeed;
        r.effectiveSpeed = nextSpeed;
        coord.decision = (reEvaluatedDecision === 'SLOW') ? 'SLOW' : 'MOVE';
        coord.finalDecision = coord.decision;
        coord.decisionReason = conflictCleared
          ? 'Conflict zone cleared; resumed movement'
          : 'Coordination re-evaluation: progressing robot released to prevent deadlock';
        coord.applied = true;
        syncDebugState(robotId);

        moveRobotToNextWaypoint(robotId, waypointIndex);
      }
    };

    coord.checkWaitResume = checkWaitResume;
    coord.waitTimerEvent = scene.time.addEvent({
      delay: 150,
      loop: true,
      callback: checkWaitResume
    });
  };

  // ── Per-waypoint tween with ML Decision Handling ─────────────────────────────
  const moveRobotToNextWaypoint = (robotId, waypointIndex) => {
    const r = robots[robotId];
    const sprite = robotSprites[robotId];
    if (!r || !sprite || r.status !== 'moving') return;

    if (!r.path || waypointIndex >= r.path.length) {
      finishRobotMovement(robotId);
      return;
    }

    r.waypointIndex = waypointIndex;

    const runMode = typeof callbacks.getRunMode === 'function' ? callbacks.getRunMode() : 'BASELINE';
    const { decision: rawDecision, isFresh, raw } = getEffectiveDecision(robotId, runMode);
    const coord = getCoordinationState(robotId);

    let activeDecision = rawDecision;

    // ── Adaptive Intervention Selection (OPTIMIZED mode only) ────────────────
    if (runMode === 'OPTIMIZED') {
      const relevantConflict = getMostImminentUpcomingConflict(robotId, waypointIndex);
      const assessment = relevantConflict
        ? (relevantConflict.temporalAssessment ||
           (typeof assessTemporalResolution === 'function' ? assessTemporalResolution(relevantConflict, robots[relevantConflict.robotA], robots[relevantConflict.robotB]) : null))
        : null;

      const otherRobotId = relevantConflict
        ? (relevantConflict.robotA === robotId ? relevantConflict.robotB : relevantConflict.robotA)
        : null;
      const otherRobot = otherRobotId ? robots[otherRobotId] : null;
      const otherCoord = otherRobotId ? getCoordinationState(otherRobotId) : null;
      const isOtherWaiting = Boolean(otherCoord && otherCoord.waiting);

      const intervention = selectAdaptiveIntervention({
        mlDecision: rawDecision,
        conflict: relevantConflict,
        temporalAssessment: assessment,
        robotState: r,
        otherRobotState: otherRobot,
        isOtherWaiting,
        consecutiveWaitCount: coord.consecutiveWaitCount || 0
      });

      activeDecision = intervention.appliedDecision;
      coord.mlDecision = intervention.mlDecision;
      const inRerouteCooldown = (Date.now() - coord.lastRerouteAt) < REROUTE_COOLDOWN_MS;
      if (inRerouteCooldown && coord.decision === 'REROUTE') {
        coord.finalDecision = 'REROUTE';
      } else {
        coord.finalDecision = intervention.finalDecision;
        coord.decisionReason = intervention.decisionReason;
      }
    }

    // ── REROUTE Handling ───────────────────────────────────────────────────────
    if (activeDecision === 'REROUTE') {
      const now = Date.now();
      const isDuplicate = raw && raw.timestamp && raw.timestamp === coord.lastReroutedDecisionTimestamp;
      const inCooldown = (now - coord.lastRerouteAt) < REROUTE_COOLDOWN_MS;

      if (isDuplicate || inCooldown) {
        // Prevent repeated rerouting on identical decision or within cooldown, but preserve REROUTE in state
        activeDecision = 'MOVE';
        coord.decision = 'REROUTE';
        coord.applied = true;
      } else if (r.destination) {
        // Ensure REROUTE is only applied when predicted conflict actually affects upcoming trajectory
        const upcomingConflictTiles = getUpcomingConflictTiles(robotId, waypointIndex);

        if (upcomingConflictTiles.length === 0) {
          // Trajectory is clear; continue on current route
          activeDecision = 'MOVE';
        } else {
          const curTileX = (map && typeof map.worldToTileX === 'function')
            ? map.worldToTileX(sprite.x)
            : Math.floor(sprite.x / 32);
          const curTileY = (map && typeof map.worldToTileY === 'function')
            ? map.worldToTileY(sprite.y)
            : Math.floor(sprite.y / 32);
          const destTileX = r.destination.tileX;
          const destTileY = r.destination.tileY;

          const alternatePath = tryCalculateAlternatePath(robotId, curTileX, curTileY, destTileX, destTileY, upcomingConflictTiles);

          if (alternatePath && alternatePath.length > 1) {
            r.path = alternatePath;
            coord.rerouteCount++;
            coord.lastRerouteAt = now;
            coord.lastReroutedDecisionTimestamp = raw?.timestamp || now;
            coord.decision = 'REROUTE';
            coord.finalDecision = 'REROUTE';
            coord.applied = true;
            coord.lastAppliedAt = now;
            syncDebugState(robotId);

            if (typeof callbacks.updatePathVisualization === 'function') {
              callbacks.updatePathVisualization(robotId);
            }
            callbacks.detectFleetConflicts();

            // Proceed on newly calculated alternate route from waypoint 1
            moveRobotToNextWaypoint(robotId, 1);
            return;
          } else {
            // Fallback: If conflict affects trajectory but no alternate path exists, WAIT
            coord.lastRerouteAt = now;
            activeDecision = 'WAIT';
          }
        }
      }
    }

    // ── WAIT Handling ──────────────────────────────────────────────────────────
    if (activeDecision === 'WAIT') {
      enterWaitState(robotId, waypointIndex, runMode);
      return;
    }

    // ── MOVE / SLOW Handling ───────────────────────────────────────────────────
    const baseSpeed = r.speed || 100;
    let effectiveSpeed = baseSpeed;

    if (activeDecision === 'SLOW') {
      // Reduce to 50% without altering r.speed in UI
      effectiveSpeed = Math.max(10, baseSpeed * 0.5);
    }

    if (coord.decision !== 'REROUTE' || (Date.now() - coord.lastRerouteAt) >= REROUTE_COOLDOWN_MS) {
      coord.decision = activeDecision;
    }
    coord.applied = true;
    coord.waiting = false;
    coord.consecutiveWaitCount = 0;
    coord.effectiveSpeed = effectiveSpeed;
    r.effectiveSpeed = effectiveSpeed;
    coord.waitStartedAt = null;
    coord.lastAppliedAt = Date.now();
    syncDebugState(robotId);

    const targetTile = r.path[waypointIndex];
    const targetWorldX = targetTile.worldX !== undefined ? targetTile.worldX : (map.tileToWorldX(targetTile.tileX) + 16);
    const targetWorldY = targetTile.worldY !== undefined ? targetTile.worldY : (map.tileToWorldY(targetTile.tileY) + 16);

    // Physical collision check (BASELINE condition)
    // If robots are already in collision contact at waypoint transition, halt immediately
    if (runMode === 'BASELINE') {
      const collision = checkPhysicalCollision(robotId, sprite.x, sprite.y, COLLISION_THRESHOLD_PX);
      if (collision) {
        handleRobotCollision(robotId, collision.otherId);
        return;
      }
    }

    // Continuous physical separation safety check (OPTIMIZED mode only)
    if (runMode === 'OPTIMIZED') {
      const obstacleAhead = checkImmediateObstacleAhead(
        robotId,
        sprite.x,
        sprite.y,
        targetWorldX,
        targetWorldY,
        PHYSICAL_STOP_THRESHOLD_PX
      );

      if (obstacleAhead) {
        // Space directly ahead is occupied within 64px physical sprite footprint; hold safely
        enterWaitState(robotId, waypointIndex, runMode);
        return;
      }

      if (activeDecision === 'MOVE') {
        const nearbyAhead = checkImmediateObstacleAhead(
          robotId,
          sprite.x,
          sprite.y,
          targetWorldX,
          targetWorldY,
          PHYSICAL_SLOW_THRESHOLD_PX
        );
        if (nearbyAhead) {
          // Smooth following velocity: reduce to 50% to prevent approaching overlap
          effectiveSpeed = Math.max(10, baseSpeed * 0.5);
          coord.effectiveSpeed = effectiveSpeed;
          r.effectiveSpeed = effectiveSpeed;
        }
      }
    }

    const distance = Math.hypot(sprite.x - targetWorldX, sprite.y - targetWorldY);
    const duration = Math.max(1, (distance / effectiveSpeed) * 1000);

    activeTweens[robotId] = scene.tweens.add({
      targets: sprite,
      x: targetWorldX,
      y: targetWorldY,
      duration: duration,
      ease: 'Linear',
      onUpdate: () => {
        r.x = sprite.x;
        r.y = sprite.y;

        // Physical collision check (BASELINE condition)
        // Detects when opposing or crossing robots make bumper contact and halts both immediately
        if (runMode === 'BASELINE') {
          const collision = checkPhysicalCollision(robotId, sprite.x, sprite.y, COLLISION_THRESHOLD_PX);
          if (collision) {
            handleRobotCollision(robotId, collision.otherId);
            return;
          }
        }

        if (runMode === 'OPTIMIZED') {
          const obstacle = checkImmediateObstacleAhead(
            robotId,
            sprite.x,
            sprite.y,
            targetWorldX,
            targetWorldY,
            PHYSICAL_STOP_THRESHOLD_PX
          );

          if (obstacle) {
            if (activeTweens[robotId]) {
              activeTweens[robotId].stop();
              activeTweens[robotId] = null;
            }
            r.x = sprite.x;
            r.y = sprite.y;
            r.start.x = sprite.x;
            r.start.y = sprite.y;
            r.previousValidPosition = { x: sprite.x, y: sprite.y };
            enterWaitState(robotId, waypointIndex, runMode);
          }
        }
      },
      onComplete: () => {
        r.x = targetWorldX;
        r.y = targetWorldY;
        r.start.x = targetWorldX;
        r.start.y = targetWorldY;
        r.start.tileX = targetTile.tileX;
        r.start.tileY = targetTile.tileY;
        if (targetTile.navX !== undefined) {
          r.start.navX = targetTile.navX;
          r.start.navY = targetTile.navY;
        }
        r.previousValidPosition = { x: targetWorldX, y: targetWorldY };

        // Step 4: Report actual pixel distance travelled
        if (typeof callbacks.onSegmentTravelled === 'function' && distance > 0) {
          callbacks.onSegmentTravelled(robotId, distance);
        }

        // Recompute fleet conflicts as robots advance past waypoints
        if (typeof callbacks.detectFleetConflicts === 'function') {
          callbacks.detectFleetConflicts();
        }

        // Immediately check and wake up waiting robots whose conflict has cleared
        for (const [otherId, otherCoord] of localCoordinationState.entries()) {
          if (otherId !== robotId && otherCoord.waiting) {
            const otherRobot = robots[otherId];
            if (otherRobot && otherRobot.status === 'moving' && typeof otherCoord.checkWaitResume === 'function') {
              otherCoord.checkWaitResume();
            }
          }
        }

        moveRobotToNextWaypoint(robotId, waypointIndex + 1);
      }
    });
  };

  // ── Start ─────────────────────────────────────────────────────────────────────
  const startRobotMovement = (robotId) => {
    const r = robots[robotId];
    const sprite = robotSprites[robotId];
    if (!r || !sprite) return;

    if (r.status === 'moving' || r.status === 'collided') return;

    if (!r.destination || !r.path || r.path.length === 0) return;

    // If already at or within same logical cell
    if (r.path.length <= 1) {
      finishRobotMovement(robotId);
      return;
    }

    const coord = getCoordinationState(robotId);
    coord.waiting = false;
    coord.waitStartedAt = null;
    if (coord.waitTimerEvent) {
      coord.waitTimerEvent.remove();
      coord.waitTimerEvent = null;
    }
    syncDebugState(robotId);

    r.status = 'moving';
    callbacks.updateStatusUI();

    const firstTargetX = r.path[0].worldX !== undefined ? r.path[0].worldX : (map.tileToWorldX(r.path[0].tileX) + 16);
    const firstTargetY = r.path[0].worldY !== undefined ? r.path[0].worldY : (map.tileToWorldY(r.path[0].tileY) + 16);
    const distToFirst = Math.hypot(sprite.x - firstTargetX, sprite.y - firstTargetY);

    let startIndex = 1;
    if (distToFirst > 4) {
      startIndex = 0;
    }

    moveRobotToNextWaypoint(robotId, startIndex);
    callbacks.detectFleetConflicts();
  };

  return {
    startRobotMovement,
    moveRobotToNextWaypoint,
    finishRobotMovement,
    resetCollisionState,
    handleRobotCollision
  };
}
