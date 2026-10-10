import 'dotenv/config';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Server as SocketIOServer } from 'socket.io';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3001);
const STALE_TIMEOUT_MS = Number(process.env.STALE_TIMEOUT_MS || 10000);

// Gemini API Configuration
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_MODEL = 'gemini-2.0-flash';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

// In-memory fleet state object: robotId -> state & features (for legacy HTTP telemetry)
const fleetState = new Map();

// Real-time Robot Registry: robotId -> { robotId, socketId, connectedAt, lastSeen, lastSeenMs, state }
const robotRegistry = new Map();

// Socket to Robot mapping: socket.id -> robotId
const socketToRobot = new Map();

// Helper to check if robot state is stale
function isRobotStale(robotId, timeoutMs = STALE_TIMEOUT_MS) {
  const record = robotRegistry.get(robotId);
  if (!record) return true;
  return (Date.now() - (record.lastSeenMs || 0)) > timeoutMs;
}

// Validator for incoming STATE_UPDATE payload
function validateStateUpdate(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return { valid: false, error: 'Payload must be a non-null object' };
  }

  // robotId exists
  if (typeof payload.robotId !== 'string' || !payload.robotId.trim()) {
    return { valid: false, error: 'Missing or invalid robotId' };
  }

  // x/y are finite numbers
  if (typeof payload.x !== 'number' || !Number.isFinite(payload.x)) {
    return { valid: false, error: 'Coordinate x must be a finite number' };
  }
  if (typeof payload.y !== 'number' || !Number.isFinite(payload.y)) {
    return { valid: false, error: 'Coordinate y must be a finite number' };
  }

  // speed is finite and non-negative
  if (typeof payload.speed !== 'number' || !Number.isFinite(payload.speed) || payload.speed < 0) {
    return { valid: false, error: 'Speed must be a non-negative finite number' };
  }

  // battery is numeric
  if (typeof payload.battery !== 'number' || !Number.isFinite(payload.battery)) {
    return { valid: false, error: 'Battery must be a numeric value' };
  }

  // priority is valid (non-empty string or finite number)
  const isPriorityValid = (typeof payload.priority === 'number' && Number.isFinite(payload.priority)) ||
                          (typeof payload.priority === 'string' && payload.priority.trim().length > 0);
  if (!isPriorityValid) {
    return { valid: false, error: 'Priority must be a valid number or non-empty string' };
  }

  // status is valid
  if (typeof payload.status !== 'string' || !payload.status.trim()) {
    return { valid: false, error: 'Status must be a non-empty string' };
  }

  // destination is valid (null, valid string, or object with finite coords if provided)
  const isDestValid = payload.destination === null ||
    (typeof payload.destination === 'string' && payload.destination.trim().length > 0) ||
    (typeof payload.destination === 'object' && payload.destination !== null &&
      (!('x' in payload.destination) || Number.isFinite(payload.destination.x)) &&
      (!('y' in payload.destination) || Number.isFinite(payload.destination.y)) &&
      (!('tileX' in payload.destination) || Number.isFinite(payload.destination.tileX)) &&
      (!('tileY' in payload.destination) || Number.isFinite(payload.destination.tileY))
    );
  if (!isDestValid) {
    return { valid: false, error: 'Destination must be null, a non-empty string, or an object with finite coordinates' };
  }

  // timestamp is valid
  const isTimestampValid = (typeof payload.timestamp === 'number' && Number.isFinite(payload.timestamp) && payload.timestamp > 0) ||
                           (typeof payload.timestamp === 'string' && !isNaN(Date.parse(payload.timestamp)));
  if (!isTimestampValid) {
    return { valid: false, error: 'Timestamp must be a positive number or valid ISO date string' };
  }

  return { valid: true };
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMINI-BASED DECISION ENGINE (replaces Python ML predictor)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build a structured Gemini prompt from 26 ARES V2 robot features.
 * The prompt instructs Gemini to act as an AMR fleet coordinator and return
 * exactly one of: MOVE, SLOW, WAIT, REROUTE with a confidence score.
 */
function buildGeminiPrompt(features) {
  return `You are an expert Autonomous Mobile Robot (AMR) fleet coordinator AI for a smart warehouse.
Your role is to decide the optimal action for a robot based on its real-time telemetry features.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation, no code fences) in this exact format:
{"decision": "ACTION", "confidence": 0.XX}

ACTION must be exactly one of: MOVE, SLOW, WAIT, REROUTE

Decision Rules (priority order — Safety >> Traffic Flow >> Efficiency):
1. SAFETY (highest priority):
   - If true_ttc_sec < 2.0 and cpa_distance_m < 1.0 → WAIT (collision imminent)
   - If closing_velocity_mps > 1.5 and nearest_robot_distance_m < 3.0 → SLOW or WAIT
   - If path_blocked = 1 → WAIT or REROUTE (never MOVE into blocked path)
   - If battery_pct < 10 and task_type is not CHARGE → consider urgency

2. TRAFFIC FLOW:
   - If obstacle_detected = 0 and nearby_robot_count = 0 and no conflicts → MOVE
   - If relative_direction = OPPOSITE and distance < 5.0 → SLOW (head-on approach)
   - If intersection_conflict = 1 → SLOW or WAIT based on priority
   - If traffic_level = CRITICAL → SLOW

3. EFFICIENCY:
   - If path is clear, no nearby robots, no conflicts → MOVE at full speed
   - If alternative_route_available = 1 and path_blocked = 1 → REROUTE
   - Prefer MOVE > SLOW > REROUTE > WAIT for efficiency when safe

Robot Telemetry Features:
- Position: (${features.current_x}, ${features.current_y}) → Destination: (${features.destination_x}, ${features.destination_y})
- Distance to destination: ${features.distance_to_destination_m}m, ETA: ${features.eta_sec}s
- Speed: ${features.speed_mps} m/s, Heading: ${features.heading_rad} rad
- Battery: ${features.battery_pct}%
- Task: ${features.task_type}, Priority: ${features.task_priority}
- Obstacle detected: ${features.obstacle_detected}, Path blocked: ${features.path_blocked}
- Alternative route available: ${features.alternative_route_available}
- Dynamic Interaction Radius: ${features.dynamic_interaction_radius_m}m
- Nearby robots: ${features.nearby_robot_count}, Traffic level: ${features.traffic_level}
- Nearest robot distance: ${features.nearest_robot_distance_m}m
- Nearest robot relative speed: ${features.nearest_robot_relative_speed_mps} m/s
- Nearest robot relative heading: ${features.nearest_robot_relative_heading_rad} rad
- Relative direction: ${features.relative_direction}
- Closing velocity: ${features.closing_velocity_mps} m/s
- True TTC: ${features.true_ttc_sec}s
- CPA time: ${features.cpa_time_sec}s, CPA distance: ${features.cpa_distance_m}m
- Intersection conflict: ${features.intersection_conflict}

Respond with ONLY the JSON object:`;
}

let geminiDisabled = false;

/**
 * Call Google Gemini API for robot decision-making.
 * Returns { decision, confidence } or fallback on error.
 */
async function getGeminiPrediction(features) {
  if (!GEMINI_API_KEY || geminiDisabled) {
    return getRuleBasedDecision(features);
  }

  try {
    const prompt = buildGeminiPrompt(features);

    const requestBody = {
      contents: [{
        parts: [{ text: prompt }]
      }],
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 60,
        topP: 0.8,
        topK: 10
      }
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(GEMINI_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const errText = await response.text();
      console.warn(`[Gemini] API error ${response.status}: ${errText.substring(0, 150)}`);
      if (response.status === 400 || response.status === 401 || response.status === 403 || response.status === 404) {
        console.warn('[Gemini] Disabling further remote calls; using rule-based coordination engine.');
        geminiDisabled = true;
      }
      return getRuleBasedDecision(features);
    }

    const data = await response.json();

    // Extract text from Gemini response
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    const cleanText = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();

    try {
      const parsed = JSON.parse(cleanText);
      const validActions = ['MOVE', 'SLOW', 'WAIT', 'REROUTE'];
      const decision = validActions.includes(parsed.decision) ? parsed.decision : 'WAIT';
      const confidence = typeof parsed.confidence === 'number'
        ? Math.max(0, Math.min(1, parsed.confidence))
        : 0.7;

      return { decision, confidence: Number(confidence.toFixed(4)) };
    } catch (parseErr) {
      // Try to extract action from text even if JSON is malformed
      const upperText = cleanText.toUpperCase();
      if (upperText.includes('MOVE')) return { decision: 'MOVE', confidence: 0.5 };
      if (upperText.includes('SLOW')) return { decision: 'SLOW', confidence: 0.5 };
      if (upperText.includes('REROUTE')) return { decision: 'REROUTE', confidence: 0.5 };
      if (upperText.includes('WAIT')) return { decision: 'WAIT', confidence: 0.5 };

      console.warn(`[Gemini] Failed to parse response: "${cleanText.substring(0, 100)}"`);
      return getRuleBasedDecision(features);
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      console.warn('[Gemini] Request timed out (5s)');
    } else {
      console.warn(`[Gemini] Request failed: ${err.message}`);
    }
    return getRuleBasedDecision(features);
  }
}

/**
 * Deterministic rule-based fallback when Gemini is unavailable.
 * Implements the same safety-first logic as the training counterfactual evaluator.
 */
function getRuleBasedDecision(f) {
  // Safety first
  if (f.path_blocked === 1) {
    if (f.alternative_route_available === 1) {
      return { decision: 'REROUTE', confidence: 0.85 };
    }
    return { decision: 'WAIT', confidence: 0.9 };
  }

  if (f.true_ttc_sec < 2.0 && f.cpa_distance_m < 1.0) {
    return { decision: 'WAIT', confidence: 0.95 };
  }

  if (f.closing_velocity_mps > 1.5 && f.nearest_robot_distance_m < 3.0) {
    return { decision: 'SLOW', confidence: 0.85 };
  }

  if (f.intersection_conflict === 1) {
    return { decision: 'SLOW', confidence: 0.8 };
  }

  // Traffic flow
  if (f.relative_direction === 'OPPOSITE' && f.nearest_robot_distance_m < 5.0) {
    return { decision: 'SLOW', confidence: 0.75 };
  }

  if (f.traffic_level === 'CRITICAL') {
    return { decision: 'SLOW', confidence: 0.7 };
  }

  // Efficiency — clear path
  if (f.nearby_robot_count === 0 && f.obstacle_detected === 0) {
    return { decision: 'MOVE', confidence: 0.95 };
  }

  if (f.nearest_robot_distance_m > 10.0) {
    return { decision: 'MOVE', confidence: 0.85 };
  }

  // Default: cautious move
  return { decision: 'MOVE', confidence: 0.6 };
}

// Helper to send JSON responses with CORS headers
function sendJson(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  res.end(JSON.stringify(data));
}

// Helper to parse JSON body from incoming request
function parseBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 1e6) {
        req.socket.destroy();
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => {
      if (!raw.trim()) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  // CORS Preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    });
    res.end();
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);

  // GET /health
  if (req.method === 'GET' && url.pathname === '/health') {
    return sendJson(res, 200, {
      status: 'ok',
      service: 'local-edge-coordinator',
      engine: 'gemini',
      model: GEMINI_MODEL,
      port: PORT,
      uptime: process.uptime(),
      fleetCount: fleetState.size,
      activeRobotsCount: robotRegistry.size,
      socketClientsCount: io.engine ? io.engine.clientsCount : 0,
      geminiConfigured: Boolean(GEMINI_API_KEY)
    });
  }

  // GET /fleet
  if (req.method === 'GET' && url.pathname === '/fleet') {
    return sendJson(res, 200, {
      fleet: Object.fromEntries(fleetState)
    });
  }

  // GET /edge/robots - inspection endpoint for real-time edge robot state
  if (req.method === 'GET' && url.pathname === '/edge/robots') {
    const now = Date.now();
    const robots = {};
    for (const [rId, record] of robotRegistry.entries()) {
      robots[rId] = {
        ...record,
        isStale: (now - (record.lastSeenMs || 0)) > STALE_TIMEOUT_MS
      };
    }
    return sendJson(res, 200, {
      count: robotRegistry.size,
      staleTimeoutMs: STALE_TIMEOUT_MS,
      robots
    });
  }

  // POST /telemetry - update fleet state
  if (req.method === 'POST' && url.pathname === '/telemetry') {
    try {
      const body = await parseBody(req);
      const robotId = body.robotId || body.id;
      if (!robotId) {
        return sendJson(res, 400, { error: 'Missing robotId in telemetry' });
      }
      fleetState.set(robotId, {
        ...body,
        updatedAt: Date.now()
      });
      return sendJson(res, 200, { status: 'recorded', robotId });
    } catch (err) {
      return sendJson(res, 400, { error: err.message });
    }
  }

  // POST /predict - predict decision using Gemini AI
  if (req.method === 'POST' && url.pathname === '/predict') {
    try {
      const body = await parseBody(req);
      const robotId = body.robotId || body.id;
      const features = body.features || body;

      // Maintain in-memory fleet state
      if (robotId) {
        fleetState.set(robotId, {
          ...features,
          updatedAt: Date.now()
        });
      }

      const result = await getGeminiPrediction(features);

      return sendJson(res, 200, {
        robotId: robotId || null,
        decision: result.decision,
        confidence: result.confidence,
        engine: 'gemini',
        ...(result.error ? { error: result.error } : {})
      });
    } catch (err) {
      return sendJson(res, 500, {
        decision: 'WAIT',
        confidence: 0.0,
        engine: 'fallback',
        error: err.message
      });
    }
  }

  return sendJson(res, 404, { error: 'Not found' });
});

// Attach Socket.IO to the HTTP server
const io = new SocketIOServer(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

io.on('connection', (socket) => {
  const initialRobotId = socket.handshake.query?.robotId || socket.handshake.auth?.robotId;
  const nowIso = new Date().toISOString();
  const nowMs = Date.now();

  if (initialRobotId && typeof initialRobotId === 'string' && initialRobotId.trim()) {
    const rId = initialRobotId.trim();
    socketToRobot.set(socket.id, rId);
    let record = robotRegistry.get(rId);
    if (!record) {
      record = {
        robotId: rId,
        socketId: socket.id,
        connectedAt: nowIso,
        lastSeen: nowIso,
        lastSeenMs: nowMs,
        state: null
      };
      robotRegistry.set(rId, record);
    } else {
      record.socketId = socket.id;
      record.lastSeen = nowIso;
      record.lastSeenMs = nowMs;
    }
  }

  // Client explicit registration event
  socket.on('REGISTER', (data, ack) => {
    try {
      const robotId = data?.robotId;
      if (!robotId || typeof robotId !== 'string' || !robotId.trim()) {
        if (typeof ack === 'function') ack({ success: false, error: 'Invalid robotId' });
        return;
      }
      const rId = robotId.trim();
      socketToRobot.set(socket.id, rId);
      const regIso = new Date().toISOString();
      const regMs = Date.now();
      let record = robotRegistry.get(rId);
      if (!record) {
        record = {
          robotId: rId,
          socketId: socket.id,
          connectedAt: regIso,
          lastSeen: regIso,
          lastSeenMs: regMs,
          state: null
        };
        robotRegistry.set(rId, record);
      } else {
        record.socketId = socket.id;
        record.lastSeen = regIso;
        record.lastSeenMs = regMs;
      }
      if (typeof ack === 'function') ack({ success: true, robotId: rId });
    } catch (err) {
      console.error('[Socket.IO] Error in REGISTER handler:', err.message);
      if (typeof ack === 'function') ack({ success: false, error: err.message });
    }
  });

  // STATE_UPDATE event
  socket.on('STATE_UPDATE', (payload, ack) => {
    try {
      const validation = validateStateUpdate(payload);
      if (!validation.valid) {
        console.warn(`[Socket.IO] Rejected malformed STATE_UPDATE from socket ${socket.id}: ${validation.error}`);
        if (typeof ack === 'function') ack({ success: false, error: validation.error });
        return;
      }

      const rId = payload.robotId.trim();
      const updateIso = new Date().toISOString();
      const updateMs = Date.now();

      socketToRobot.set(socket.id, rId);

      let record = robotRegistry.get(rId);
      if (!record) {
        record = {
          robotId: rId,
          socketId: socket.id,
          connectedAt: updateIso,
          lastSeen: updateIso,
          lastSeenMs: updateMs,
          state: null
        };
        robotRegistry.set(rId, record);
      } else {
        record.socketId = socket.id;
        record.lastSeen = updateIso;
        record.lastSeenMs = updateMs;
      }

      // Store isolated state in registry
      record.state = {
        robotId: rId,
        x: payload.x,
        y: payload.y,
        speed: payload.speed,
        heading: payload.heading !== undefined ? payload.heading : 0,
        battery: payload.battery,
        task: payload.task !== undefined ? payload.task : null,
        priority: payload.priority,
        status: payload.status,
        destination: payload.destination,
        timestamp: payload.timestamp
      };

      // Broadcast received state to connected clients as NEIGHBOUR_STATE
      socket.broadcast.emit('NEIGHBOUR_STATE', record.state);

      if (typeof ack === 'function') {
        ack({ success: true, robotId: rId });
      }
    } catch (err) {
      console.error(`[Socket.IO] Error handling STATE_UPDATE from socket ${socket.id}:`, err.message);
      if (typeof ack === 'function') {
        ack({ success: false, error: 'Internal server error processing state update' });
      }
    }
  });

  // HEARTBEAT event
  socket.on('HEARTBEAT', (data, ack) => {
    try {
      const rId = data?.robotId || socketToRobot.get(socket.id);
      if (rId && robotRegistry.has(rId)) {
        const record = robotRegistry.get(rId);
        record.lastSeen = new Date().toISOString();
        record.lastSeenMs = Date.now();
        if (typeof ack === 'function') ack({ success: true, robotId: rId, lastSeen: record.lastSeen });
      } else if (typeof ack === 'function') {
        ack({ success: false, error: 'Robot not registered' });
      }
    } catch (err) {
      console.error('[Socket.IO] Error handling HEARTBEAT:', err.message);
      if (typeof ack === 'function') ack({ success: false, error: err.message });
    }
  });

  // DISCONNECT handling
  socket.on('disconnect', (reason) => {
    try {
      const robotId = socketToRobot.get(socket.id);
      if (robotId) {
        robotRegistry.delete(robotId);
        socketToRobot.delete(socket.id);
        io.emit('ROBOT_DISCONNECTED', { robotId });
        console.log(`[Socket.IO] Robot disconnected: ${robotId} (${reason})`);
      }
    } catch (err) {
      console.error('[Socket.IO] Error handling disconnect:', err.message);
    }
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[EDGE-COORDINATOR] Server running on http://127.0.0.1:${PORT} (Socket.IO enabled)`);
  console.log(`[EDGE-COORDINATOR] Decision engine: Gemini (${GEMINI_MODEL})`);
  console.log(`[EDGE-COORDINATOR] Gemini API key: ${GEMINI_API_KEY ? 'configured ✓' : 'NOT SET — using rule-based fallback'}`);
});

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('SIGTERM', () => {
  process.exit(0);
});

export { server, io, robotRegistry, socketToRobot, validateStateUpdate, isRobotStale, STALE_TIMEOUT_MS };
