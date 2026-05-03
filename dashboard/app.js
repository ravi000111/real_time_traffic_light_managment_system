/* =========================================================
   app.js  –  Traffic Light Management Dashboard
   Simulates the same logic as tlm_test.py:
     • 3 lanes, proportional green time allocation
     • Fixed cycle budget 120 s, yellow 3 s
     • Min green 10 s, max green 60 s
   ========================================================= */

// ── Constants (mirror tlm_test.py) ──────────────────────────
const CYCLE_BUDGET = 120;   // seconds
const YELLOW       = 3;     // seconds
const MIN_GREEN    = 10;
const MAX_GREEN    = 60;
const NUM_LANES    = 3;

// ── Simulated vehicle counts (randomised each cycle) ─────────
let vehicleCounts = [0, 0, 0];
let cycleNumber   = 1;
let schedule      = [];          // [{green, red}, …]
let phaseIndex    = 0;           // which lane is currently GREEN
let phaseState    = 'GREEN';     // GREEN | YELLOW | RED
let phaseRemaining = 0;
let totalElapsed   = 0;

// ── Build a new cycle schedule ────────────────────────────────
function buildSchedule() {
  // Pick random vehicle counts for demo
  vehicleCounts = [
    Math.floor(Math.random() * 25) + 2,
    Math.floor(Math.random() * 25) + 2,
    Math.floor(Math.random() * 25) + 2
  ];

  const totalVehicles = vehicleCounts.reduce((a, b) => a + b, 0) || 1;

  // Available green budget (subtract all yellow phases)
  const greenBudget = CYCLE_BUDGET - NUM_LANES * YELLOW;

  schedule = vehicleCounts.map(vc => {
    let g = Math.round((vc / totalVehicles) * greenBudget);
    g = Math.max(MIN_GREEN, Math.min(MAX_GREEN, g));
    const r = CYCLE_BUDGET - g - YELLOW;
    return { green: g, yellow: YELLOW, red: Math.max(0, r) };
  });

  // Update vehicle count UI
  const maxVehicles = Math.max(...vehicleCounts, 1);
  for (let i = 0; i < NUM_LANES; i++) {
    document.getElementById(`count-${i}`).textContent = vehicleCounts[i];
    const pct = Math.round((vehicleCounts[i] / maxVehicles) * 100);
    document.getElementById(`vcbar-${i}`).style.width = pct + '%';
    updatePlanBar(i, schedule[i]);
  }

  log(`🔄 Cycle #${cycleNumber} — Vehicles: [${vehicleCounts.join(', ')}]  Green: [${schedule.map(s => s.green + 's').join(', ')}]`, 'info');
}

// ── Update plan bar segments ──────────────────────────────────
function updatePlanBar(lane, plan) {
  const total = plan.green + plan.yellow + plan.red;
  const gPct  = (plan.green  / total * 100).toFixed(1);
  const yPct  = (plan.yellow / total * 100).toFixed(1);
  const rPct  = (plan.red    / total * 100).toFixed(1);

  document.getElementById(`pseg-${lane}-g`).style.width = gPct + '%';
  document.getElementById(`pseg-${lane}-y`).style.width = yPct + '%';
  document.getElementById(`pseg-${lane}-r`).style.width = rPct + '%';

  document.getElementById(`plabel-${lane}-g`).textContent = `G: ${plan.green}s`;
  document.getElementById(`plabel-${lane}-y`).textContent = `Y: ${plan.yellow}s`;
  document.getElementById(`plabel-${lane}-r`).textContent = `R: ${plan.red}s`;
}

// ── Apply signal state to DOM ─────────────────────────────────
function applySignals(greenLane, state) {
  for (let i = 0; i < NUM_LANES; i++) {
    const card = document.getElementById(`lane-${i}`);
    const stateBadge = document.getElementById(`state-${i}`);
    const timer      = document.getElementById(`timer-${i}`);
    const lR = document.getElementById(`lamp-${i}-red`);
    const lY = document.getElementById(`lamp-${i}-yellow`);
    const lG = document.getElementById(`lamp-${i}-green`);

    // Reset lamps
    [lR, lY, lG].forEach(l => l.classList.remove('active'));
    card.classList.remove('active-green', 'active-yellow', 'active-red');

    if (i === greenLane) {
      if (state === 'GREEN') {
        lG.classList.add('active');
        card.classList.add('active-green');
        stateBadge.className = 'state-badge green';
        stateBadge.textContent = 'GREEN';
        timer.style.color = 'var(--green)';
      } else if (state === 'YELLOW') {
        lY.classList.add('active');
        card.classList.add('active-yellow');
        stateBadge.className = 'state-badge yellow';
        stateBadge.textContent = 'YELLOW';
        timer.style.color = 'var(--yellow)';
      }
    } else {
      lR.classList.add('active');
      card.classList.add('active-red');
      stateBadge.className = 'state-badge red';
      stateBadge.textContent = 'RED';
      timer.style.color = 'var(--text-muted)';
    }
  }
}

// ── Update countdown timers ───────────────────────────────────
function updateTimers() {
  for (let i = 0; i < NUM_LANES; i++) {
    const timer = document.getElementById(`timer-${i}`);
    if (i === phaseIndex) {
      timer.textContent = phaseRemaining + 's';
    } else {
      timer.textContent = '—';
    }
  }
}

// ── Log helper ────────────────────────────────────────────────
function log(msg, type = '') {
  const body = document.getElementById('logBody');
  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const now = new Date();
  const ts  = now.toTimeString().split(' ')[0];

  entry.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg ${type}">${msg}</span>`;
  body.appendChild(entry);
  body.scrollTop = body.scrollHeight;

  // Cap log at 200 entries
  while (body.children.length > 200) body.removeChild(body.firstChild);
}

function clearLog() {
  document.getElementById('logBody').innerHTML = '';
}

// ── Clock ─────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clock').textContent = new Date().toTimeString().split(' ')[0];
}

// ── Main tick (1 s interval) ──────────────────────────────────
function tick() {
  if (phaseRemaining <= 0) {
    // Advance state machine
    if (phaseState === 'GREEN') {
      phaseState     = 'YELLOW';
      phaseRemaining = YELLOW;
      log(`⚠️  Lane ${phaseIndex} → YELLOW`, 'yellow');
    } else if (phaseState === 'YELLOW') {
      phaseState = 'RED-TRANSITION';
      log(`🔴 Lane ${phaseIndex} → RED`, 'red');

      // Move to next lane
      phaseIndex = (phaseIndex + 1) % NUM_LANES;

      if (phaseIndex === 0) {
        // New cycle
        cycleNumber++;
        document.getElementById('cycleNum').textContent = cycleNumber;
        buildSchedule();
      }

      phaseState     = 'GREEN';
      phaseRemaining = schedule[phaseIndex].green;
      log(`🟢 Lane ${phaseIndex} → GREEN (${phaseRemaining}s)`, 'green');
    }
  }

  applySignals(phaseIndex, phaseState);
  updateTimers();
  phaseRemaining--;
  totalElapsed++;
}

// ── Initialise ────────────────────────────────────────────────
function init() {
  // Detect device label (just a UI label)
  document.getElementById('deviceLabel').textContent =
    navigator.userAgent.includes('Win') ? 'CUDA' : 'CPU';

  buildSchedule();

  phaseIndex     = 0;
  phaseState     = 'GREEN';
  phaseRemaining = schedule[0].green;

  log(`✅ System initialised — YOLOv8 custom weights loaded`, 'info');
  log(`📷 3 camera feeds active`, 'info');
  log(`🟢 Lane 0 → GREEN (${phaseRemaining}s)`, 'green');

  applySignals(phaseIndex, phaseState);
  updateTimers();

  setInterval(tick, 1000);
  setInterval(updateClock, 1000);
  updateClock();
}

window.addEventListener('DOMContentLoaded', init);
