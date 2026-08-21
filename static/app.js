const pieces = {
  p: "&#9823;",
  r: "&#9820;",
  n: "&#9822;",
  b: "&#9821;",
  q: "&#9819;",
  k: "&#9818;",
  P: "&#9817;",
  R: "&#9814;",
  N: "&#9816;",
  B: "&#9815;",
  Q: "&#9813;",
  K: "&#9812;",
};

const config = window.IITI_CONFIG || {};
const ACTIVE_GAME_STORAGE_KEY = "rlchess.activeGameId";
const AUTH_TOKEN_STORAGE_KEY = "rlchess.authToken";
const AUTH_USER_STORAGE_KEY = "rlchess.authUser";
const GUEST_MODE_STORAGE_KEY = "rlchess.guestMode";
const JUST_SIGNED_IN_KEY = "rlchess.justSignedIn";
const authState = {
  token: localStorage.getItem(AUTH_TOKEN_STORAGE_KEY),
  user: JSON.parse(localStorage.getItem(AUTH_USER_STORAGE_KEY) || "null"),
};
const welcomeOverlay = document.querySelector("#welcomeOverlay");
const guestEntry = document.querySelector("#guestEntry");
const accountStatus = document.querySelector("#accountStatus");
const welcomeAccessTitle = document.querySelector("#welcomeAccessTitle");
const welcomeAccessDescription = document.querySelector("#welcomeAccessDescription");
function enterChess() {
  welcomeOverlay?.classList.add("hidden");
}
if (sessionStorage.getItem(JUST_SIGNED_IN_KEY) === "true") {
  sessionStorage.removeItem(JUST_SIGNED_IN_KEY);
  enterChess();
}
guestEntry?.addEventListener("click", (event) => {
  event.stopPropagation();
  if (!authState.user) {
    authState.token = null;
    authState.user = null;
    localStorage.setItem(GUEST_MODE_STORAGE_KEY, "true");
    localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    localStorage.removeItem(AUTH_USER_STORAGE_KEY);
    document.querySelector("#playerName").value = "Guest";
    updateLoginStatus();
    setupGoogleLogin();
  }
  enterChess();
});

const qualityMeta = {
  excellent: { label: "Excellent", symbol: "!!" },
  good: { label: "Good", symbol: "!" },
  book: { label: "Book", symbol: "book" },
  inaccuracy: { label: "Inaccuracy", symbol: "?!" },
  mistake: { label: "Mistake", symbol: "?" },
  blunder: { label: "Blunder", symbol: "??" },
};

const state = {
  gameId: null,
  fen: "startpos",
  legalMoves: [],
  history: [],
  selected: null,
  flipped: false,
  lastMove: null,
  gameOver: false,
  report: null,
  difficulty: "medium",
  canMove: false,
  pendingEngine: false,
};

const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const historyEl = document.querySelector("#history");
const analysisEl = document.querySelector("#analysis");
const chatLog = document.querySelector("#chatLog");
const chatInput = document.querySelector("#chatInput");
const chatSend = document.querySelector("#chatSend");
const uciInput = document.querySelector("#uciInput");
const analyzeButton = document.querySelector("#analyzeButton");
const resignButton = document.querySelector("#resignButton");
const newGameButton = document.querySelector("#newGameFooter");
const undoButton = document.querySelector("#undoButton");
const redoButton = document.querySelector("#redoButton");
const depthInput = document.querySelector("#searchDepth");
const timeInput = document.querySelector("#moveTime");
const googleButton = document.querySelector("#googleButton");
const googleSlot = document.querySelector(".google-slot");
const googleFallback = document.querySelector("#googleFallback");
const accountGoogleButton = document.querySelector("#accountGoogleButton");
const accountGoogleFallback = document.querySelector("#accountGoogleFallback");
const loginStatus = document.querySelector("#loginStatus");
const accountGoogleSlot = document.querySelector(".account-google-slot");
const loginPanel = document.querySelector("#loginPanel");
const gameResult = document.querySelector("#gameResult");
const gameResultTitle = document.querySelector("#gameResultTitle");
const gameResultDetail = document.querySelector("#gameResultDetail");
let googleButtonsRendered = false;

[googleFallback, accountGoogleFallback].filter(Boolean).forEach((button) => {
  button.addEventListener("click", () => {
    if (window.google?.accounts?.id) window.google.accounts.id.prompt();
  });
});

function apiUrl(path) {
  return `${config.API_BASE_URL || ""}${path}`;
}

function updateLoginStatus() {
  if (authState.user) {
    googleSlot?.classList.add("hidden");
    accountGoogleSlot?.classList.add("hidden");
    loginPanel?.classList.add("hidden");
    if (guestEntry) guestEntry.textContent = "Continue to Chess";
    document.querySelector("#playerName").value = authState.user.name;
    document.querySelector("#playerLabel").textContent = authState.user.name;
    welcomeAccessTitle.textContent = "Signed in through Google";
    welcomeAccessDescription.textContent = "Your personal Elo is saved to this Google account.";
    loginStatus.textContent = `Signed in through Google as ${authState.user.name}.`;
    accountStatus.textContent = `Signed in as ${authState.user.name}. Elo is saved to this account.`;
  } else if (!config.GOOGLE_CLIENT_ID) {
    googleSlot?.classList.remove("hidden");
    accountGoogleSlot?.classList.remove("hidden");
    loginPanel?.classList.remove("hidden");
    if (guestEntry) guestEntry.textContent = "Enter as Guest";
    document.querySelector("#playerName").value = "Guest";
    welcomeAccessTitle.textContent = "Joined as Guest";
    welcomeAccessDescription.textContent = "Google sign-in is not configured yet. Guest play is available.";
    loginStatus.textContent = "Joined as Guest. Google login is not configured yet.";
    accountStatus.textContent = "Google login is not configured yet. Guest play is available.";
  } else if (localStorage.getItem(GUEST_MODE_STORAGE_KEY) === "true") {
    googleSlot?.classList.remove("hidden");
    accountGoogleSlot?.classList.remove("hidden");
    loginPanel?.classList.remove("hidden");
    if (guestEntry) guestEntry.textContent = "Continue as Guest";
    document.querySelector("#playerName").value = "Guest";
    welcomeAccessTitle.textContent = "Joined as Guest";
    welcomeAccessDescription.textContent = "Sign in with Google to save your Elo across games.";
    loginStatus.textContent = "Joined as Guest. Sign in with Google to save your Elo.";
    accountStatus.textContent = "Joined as Guest. Sign in with Google to save your Elo.";
  } else {
    googleSlot?.classList.remove("hidden");
    accountGoogleSlot?.classList.remove("hidden");
    loginPanel?.classList.remove("hidden");
    if (guestEntry) guestEntry.textContent = "Enter as Guest";
    document.querySelector("#playerName").value = "Guest";
    welcomeAccessTitle.textContent = "Save your rating";
    welcomeAccessDescription.textContent = "Sign in with Google to keep your personal Elo across games.";
    loginStatus.textContent = "Sign in with Google or continue as a guest.";
    accountStatus.textContent = "Sign in with Google to save your Elo, or continue as a guest.";
  }
}

async function handleGoogleCredential(response) {
  try {
    const payload = await postJson("/api/auth/google", { credential: response.credential });
    authState.token = payload.token;
    authState.user = payload.user;
    sessionStorage.setItem(JUST_SIGNED_IN_KEY, "true");
    localStorage.removeItem(GUEST_MODE_STORAGE_KEY);
    localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, payload.token);
    localStorage.setItem(AUTH_USER_STORAGE_KEY, JSON.stringify(payload.user));
    document.querySelector("#playerName").value = payload.user.name;
    updateLoginStatus();
    enterChess();
  } catch (error) {
    loginStatus.textContent = `Sign-in failed: ${error.message}`;
  }
}

function setupGoogleLogin() {
  updateLoginStatus();
  if (authState.user || googleButtonsRendered) return;
  const googleButtons = [googleButton, accountGoogleButton].filter(Boolean);
  if (!googleButtons.length) return;
  if (!config.GOOGLE_CLIENT_ID) {
    googleButtons.forEach((button) => {
      button.textContent = "Configure Google login";
      button.classList.add("google-disabled");
    });
    [googleFallback, accountGoogleFallback].filter(Boolean).forEach((button) => {
      button.textContent = "Google login needs configuration";
      button.disabled = true;
    });
    return;
  }
  let attempts = 0;
  const render = () => {
    if (window.google?.accounts?.id) {
      window.google.accounts.id.initialize({
        client_id: config.GOOGLE_CLIENT_ID,
        callback: handleGoogleCredential,
      });
      googleButtons.forEach((button) => {
        window.google.accounts.id.renderButton(button, {
          theme: "outline",
          size: "medium",
          text: "signin_with",
          shape: "rectangular",
          width: 260,
        });
      });
      [googleFallback, accountGoogleFallback].filter(Boolean).forEach((button) => button.classList.add("hidden"));
      googleButtonsRendered = true;
      return;
    }
    if (attempts++ < 30) {
      window.setTimeout(render, 200);
    } else {
      googleButtons.forEach((button) => {
        button.textContent = "Google sign-in script unavailable";
        button.classList.add("google-disabled");
      });
      [googleFallback, accountGoogleFallback].filter(Boolean).forEach((button) => {
        button.textContent = "Google sign-in unavailable";
        button.disabled = true;
      });
    }
  };
  render();
}

const presets = {
  easy: { depth: 3, time: 1.0 },
  medium: { depth: 5, time: 2.0 },
  hard: { depth: 7, time: 3.5 },
};

function files() {
  return state.flipped ? ["h", "g", "f", "e", "d", "c", "b", "a"] : ["a", "b", "c", "d", "e", "f", "g", "h"];
}

function ranks() {
  return state.flipped ? [1, 2, 3, 4, 5, 6, 7, 8] : [8, 7, 6, 5, 4, 3, 2, 1];
}

function fenMap(fen) {
  const map = {};
  const placement = fen === "startpos" ? "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" : fen.split(" ")[0];
  placement.split("/").forEach((row, rowIndex) => {
    let fileIndex = 0;
    for (const char of row) {
      const empty = Number(char);
      if (Number.isInteger(empty) && empty > 0) {
        fileIndex += empty;
      } else {
        map[`${"abcdefgh"[fileIndex]}${8 - rowIndex}`] = char;
        fileIndex += 1;
      }
    }
  });
  return map;
}

function fenFromMap(map, turn) {
  const rows = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    let row = "";
    let empty = 0;
    for (const file of "abcdefgh") {
      const piece = map[`${file}${rank}`];
      if (!piece) {
        empty += 1;
      } else {
        if (empty) row += empty;
        row += piece;
        empty = 0;
      }
    }
    if (empty) row += empty;
    rows.push(row);
  }
  return `${rows.join("/")} ${turn} - - 0 1`;
}

function previewMove(move) {
  if (!move || move.length < 4) return;
  const map = fenMap(state.fen);
  const from = move.slice(0, 2);
  const to = move.slice(2, 4);
  const promotion = move[4];
  const piece = map[from];
  if (!piece) return;

  delete map[from];
  map[to] = promotion
    ? (piece === piece.toUpperCase() ? promotion.toUpperCase() : promotion)
    : piece;

  if (piece.toLowerCase() === "k" && from === "e1" && to === "g1") {
    map.f1 = map.h1;
    delete map.h1;
  }
  if (piece.toLowerCase() === "k" && from === "e1" && to === "c1") {
    map.d1 = map.a1;
    delete map.a1;
  }
  if (piece.toLowerCase() === "k" && from === "e8" && to === "g8") {
    map.f8 = map.h8;
    delete map.h8;
  }
  if (piece.toLowerCase() === "k" && from === "e8" && to === "c8") {
    map.d8 = map.a8;
    delete map.a8;
  }

  state.fen = fenFromMap(map, state.fen.includes(" w ") ? "b" : "w");
  state.lastMove = move;
  state.legalMoves = [];
  renderBoard();
}

function renderBoard() {
  const map = fenMap(state.fen);
  const moveSquares = state.lastMove ? [state.lastMove.slice(0, 2), state.lastMove.slice(2, 4)] : [];
  boardEl.innerHTML = "";

  for (const rank of ranks()) {
    for (const file of files()) {
      const square = `${file}${rank}`;
      const button = document.createElement("button");
      const isDark = (file.charCodeAt(0) - 97 + rank) % 2 === 0;
      button.className = `square ${isDark ? "dark" : "light"}`;
      if (state.selected === square) button.classList.add("selected");
      if (moveSquares.includes(square)) button.classList.add("last-move");
      if (state.selected && state.legalMoves.some((move) => move.startsWith(state.selected + square))) {
        button.classList.add("legal");
      }
      const piece = map[square];
      button.dataset.square = square;
      button.innerHTML = `<span class="piece ${piece === piece?.toUpperCase() ? "white-piece" : "black-piece"}">${pieces[piece] || ""}</span><span class="coord">${square}</span>`;
      button.addEventListener("click", () => chooseSquare(square));
      boardEl.appendChild(button);
    }
  }
}

function chooseSquare(square) {
  if (!state.gameId || state.gameOver || !state.canMove || state.pendingEngine) return;
  const map = fenMap(state.fen);
  if (!state.selected) {
    const piece = map[square];
    if (!piece) return;
    if (!state.legalMoves.some((move) => move.startsWith(square))) return;
    state.selected = square;
    renderBoard();
    return;
  }

  const promotion = state.legalMoves.find((move) => move.startsWith(state.selected + square) && move.length === 5);
  const move = promotion || `${state.selected}${square}`;
  if (!state.legalMoves.includes(move)) {
    if (map[square] && state.legalMoves.some((legalMove) => legalMove.startsWith(square))) {
      state.selected = square;
    } else {
      state.selected = null;
    }
    renderBoard();
    return;
  }
  state.selected = null;
  sendMove(move);
}

function applyPayload(payload) {
  state.gameId = payload.game_id;
  if (state.gameId) localStorage.setItem(ACTIVE_GAME_STORAGE_KEY, state.gameId);
  state.fen = payload.fen;
  state.legalMoves = payload.legal_moves || [];
  state.history = payload.history || [];
  state.lastMove = payload.last_move || payload.engine_move || state.lastMove;
  state.gameOver = Boolean(payload.game_over || payload.result);
  state.canMove = Boolean(payload.can_move);
  state.pendingEngine = false;
  const visiblePlayerName = authState.user?.name || payload.account?.name;
  if (visiblePlayerName) {
    document.querySelector("#playerName").value = visiblePlayerName;
    document.querySelector("#playerLabel").textContent = visiblePlayerName;
  }

  statusEl.textContent = state.gameOver
    ? payload.status
    : state.canMove
      ? payload.status
      : `${payload.status} Waiting for engine.`;
  if (state.gameOver) {
    gameResultTitle.textContent = payload.status || "Game over";
    gameResultDetail.textContent = payload.result
      ? `Result: ${payload.result}`
      : "Start a new game when ready.";
    gameResult.classList.remove("hidden");
  } else {
    gameResult.classList.add("hidden");
    gameResultDetail.textContent = "";
  }
  document.querySelector("#turnLabel").textContent = payload.turn || "--";
  document.querySelector("#engineState").textContent = payload.engine_status || "RL Chess Python Engine · approx. 2000 Elo";
  document.querySelector("#engineMeta").textContent = payload.engine_details || "NNUE HalfKP · incremental accumulators · NumPy inference";
  if (payload.player) {
    document.querySelector("#playerElo").textContent = payload.player.games_played
      ? `Elo ${payload.player.elo}`
      : "Rating not established";
  }
  analyzeButton.disabled = !state.gameOver;
  analyzeButton.classList.toggle("locked", !state.gameOver);
  resignButton.disabled = state.gameOver || !state.gameId;
  resignButton.classList.toggle("active-resign", Boolean(state.gameId && !state.gameOver));
  newGameButton.disabled = false;
  newGameButton.classList.remove("locked");
  undoButton.disabled = !Boolean(payload.can_undo) || state.pendingEngine;
  redoButton.disabled = !Boolean(payload.can_redo) || state.pendingEngine;
  analyzeButton.textContent = state.gameOver ? "Analyze" : "Analyze";

  renderBoard();
  renderHistory();
  renderAnalysis(payload.analysis || []);
}

function renderHistory() {
  historyEl.innerHTML = "";
  document.querySelector("#moveCount").textContent = `${state.history.length} moves`;
  state.history.forEach((item, index) => {
    const li = document.createElement("li");
    const classification = item.classification;
    const badge = classification ? `<span class="quality ${classification.tone}">${classification.symbol}</span>` : "";
    li.innerHTML = `<span>${index + 1}. ${item.side}</span><strong>${item.san}</strong>${badge}`;
    historyEl.appendChild(li);
  });
}

function renderAnalysis(lines) {
  analysisEl.innerHTML = "";
  if (!lines.length) {
    analysisEl.textContent = "Engine search info appears after a move.";
    return;
  }
  const line = lines[0];
  const row = document.createElement("div");
  row.className = "line";
  row.innerHTML = `<strong>${line.move || "--"}</strong><span>depth ${line.depth} | ${line.nodes} nodes | ${line.nps} nps</span><span>score ${Number(line.score).toFixed(2)}</span>`;
  analysisEl.appendChild(row);
}

function renderPersonalization(personalization) {
  const data = personalization || {};
  const recommendation = data.recommendation || {};
  document.querySelector("#personalizationGames").textContent = `${data.games_analyzed || 0} past matches`;
  document.querySelector("#personalizationSummary").textContent = data.history_text || "Personalized coaching will appear after more analyzed matches.";

  const recommendationEl = document.querySelector("#settingRecommendation");
  recommendationEl.innerHTML = "";
  if (recommendation.depth) {
    recommendationEl.innerHTML = `<div><span>Recommended next setting</span><strong>${recommendation.difficulty} · depth ${recommendation.depth} · ${Number(recommendation.move_time).toFixed(1)}s</strong><small>${recommendation.reason}</small></div>`;
  }

  const patternsEl = document.querySelector("#personalizationPatterns");
  patternsEl.innerHTML = "";
  (data.patterns || []).forEach((pattern) => {
    const item = document.createElement("article");
    item.className = `personalization-pattern ${pattern.tone || "inaccuracy"}`;
    item.innerHTML = `<strong>${pattern.title}</strong><p>${pattern.detail}</p>`;
    patternsEl.appendChild(item);
  });
}

function renderReport(report) {
  state.report = report;
  document.querySelector("#playView").classList.remove("active");
  document.querySelector("#reviewView").classList.add("active");
  document.querySelector("#accuracy").textContent = `${report.accuracy}%`;
  document.querySelector("#verdict").textContent = report.verdict;
  document.querySelector("#resultLabel").textContent = report.result || "live";
  document.querySelector("#pgnBox").textContent = report.pgn || "";
  renderPersonalization(report.personalization);
  if (report.player) {
    document.querySelector("#playerElo").textContent = report.player.games_played
      ? `Elo ${report.player.elo}`
      : "Rating not established";
  }

  const scoreBars = document.querySelector("#scoreBars");
  scoreBars.innerHTML = "";
  Object.entries(report.counts).forEach(([tone, count]) => {
    const meta = qualityMeta[tone] || { label: tone, symbol: "" };
    const bar = document.createElement("div");
    bar.className = `score-card ${tone}`;
    bar.innerHTML = `<span>${meta.symbol}</span><strong>${count}</strong><small>${meta.label}</small>`;
    scoreBars.appendChild(bar);
  });

  const reviewMoves = document.querySelector("#reviewMoves");
  reviewMoves.innerHTML = "";
  report.moves.forEach((item, index) => {
    const classification = item.classification;
    const row = document.createElement("article");
    row.className = `review-row ${classification.tone}`;
    row.innerHTML = `
      <div class="review-symbol">${classification.symbol}</div>
      <div>
        <strong>${index + 1}. ${item.san} - ${classification.label}</strong>
        <p>${classification.reason}</p>
        ${classification.best_move ? `<small>Engine preferred ${classification.best_move}; played ${classification.played_move}</small>` : ""}
      </div>
    `;
    reviewMoves.appendChild(row);
  });
}

async function postJson(url, body) {
  const headers = { "Content-Type": "application/json" };
  if (authState.token) headers.Authorization = `Bearer ${authState.token}`;
  const response = await fetch(apiUrl(url), {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || `HTTP ${response.status}`);
  }
  if (!response.ok) throw new Error(data.detail || text || "Request failed");
  return data;
}

async function getJson(url) {
  const response = await fetch(apiUrl(url));
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || `HTTP ${response.status}`);
  }
  if (!response.ok) throw new Error(data.detail || text || "Request failed");
  return data;
}

async function resumeSavedGame() {
  const savedGameId = localStorage.getItem(ACTIVE_GAME_STORAGE_KEY);
  if (!savedGameId) return;
  try {
    const payload = await getJson(`/api/game/${encodeURIComponent(savedGameId)}`);
    applyPayload(payload);
  } catch {
    localStorage.removeItem(ACTIVE_GAME_STORAGE_KEY);
  }
}

async function startGame(event) {
  event?.preventDefault();
  if (state.gameId && !state.gameOver) {
    statusEl.textContent = "Resign or finish the current game before starting a new one.";
    return;
  }
  try {
    document.querySelector("#playView").classList.add("active");
    document.querySelector("#reviewView").classList.remove("active");
    document.querySelector("#playerLabel").textContent = document.querySelector("#playerName").value || "Guest";
    state.report = null;
    state.gameId = null;
    localStorage.removeItem(ACTIVE_GAME_STORAGE_KEY);
    state.fen = "startpos";
    state.history = [];
    state.selected = null;
    state.lastMove = null;
    state.gameOver = false;
    state.canMove = false;
    state.pendingEngine = false;
    newGameButton.disabled = true;
    const payload = await postJson("/api/new-game", {
      player_name: document.querySelector("#playerName").value,
      player_color: document.querySelector("#playerColor").value,
      difficulty: state.difficulty,
      search_depth: Number(depthInput.value),
      move_time: Number(timeInput.value),
      player_token: authState.token,
    });
    applyPayload(payload);
  } catch (error) {
    newGameButton.disabled = false;
    statusEl.textContent = error.message;
  }
}

async function sendMove(move) {
  if (!state.gameId || !move || !state.canMove || state.pendingEngine) return;
  let previousFen = state.fen;
  let previousLegalMoves = state.legalMoves;
  try {
    state.pendingEngine = true;
    state.canMove = false;
    previewMove(move);
    statusEl.textContent = "Engine thinking...";
    const payload = await postJson("/api/move", { game_id: state.gameId, move });
    uciInput.value = "";
    applyPayload(payload);
  } catch (error) {
    state.pendingEngine = false;
    state.canMove = true;
    if (typeof previousFen !== "undefined") state.fen = previousFen;
    if (typeof previousLegalMoves !== "undefined") state.legalMoves = previousLegalMoves;
    renderBoard();
    statusEl.textContent = error.message;
  }
}

async function analyzeGame() {
  if (!state.gameId || !state.gameOver) return;
  try {
    analyzeButton.disabled = true;
    analyzeButton.classList.add("thinking");
    analyzeButton.textContent = "Analyzing...";
    statusEl.textContent = "Analyzing game...";
    const report = state.report || await postJson("/api/analyze", { game_id: state.gameId });
    analyzeButton.classList.remove("thinking");
    analyzeButton.textContent = "Analyze";
    analyzeButton.disabled = false;
    renderReport(report);
  } catch (error) {
    analyzeButton.classList.remove("thinking");
    analyzeButton.textContent = "Analyze";
    analyzeButton.disabled = false;
    statusEl.textContent = error.message;
  }
}

function appendChat(role, message) {
  const item = document.createElement("div");
  item.className = `chat-message ${role}`;
  item.textContent = message;
  chatLog.appendChild(item);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendChat() {
  const message = chatInput.value.trim();
  if (!message || !state.gameId) return;
  chatInput.value = "";
  appendChat("user", message);
  chatSend.disabled = true;
  appendChat("thinking", "Thinking...");
  const thinking = chatLog.lastElementChild;
  try {
    const payload = await postJson("/api/chat", { game_id: state.gameId, message });
    thinking.remove();
    appendChat("assistant", payload.text);
  } catch (error) {
    thinking.remove();
    appendChat("error", error.message);
  } finally {
    chatSend.disabled = false;
  }
}

async function undoMove() {
  if (!state.gameId || state.gameOver || state.pendingEngine) return;
  try {
    statusEl.textContent = "Undoing last player turn...";
    const payload = await postJson("/api/undo", { game_id: state.gameId });
    applyPayload(payload);
  } catch (error) {
    statusEl.textContent = error.message;
  }
}

async function redoMove() {
  if (!state.gameId || state.gameOver || state.pendingEngine) return;
  try {
    statusEl.textContent = "Redoing move...";
    const payload = await postJson("/api/redo", { game_id: state.gameId });
    applyPayload(payload);
  } catch (error) {
    statusEl.textContent = error.message;
  }
}

async function resignGame() {
  if (!state.gameId || state.gameOver || state.pendingEngine) return;
  try {
    statusEl.textContent = "Resigning game...";
    const payload = await postJson("/api/resign", { game_id: state.gameId });
    applyPayload(payload);
  } catch (error) {
    statusEl.textContent = error.message;
  }
}

document.querySelector("#newGameForm").addEventListener("submit", startGame);
newGameButton.addEventListener("click", () => startGame());
document.querySelector("#sendMove").addEventListener("click", () => sendMove(uciInput.value.trim()));
document.querySelector("#analyzeButton").addEventListener("click", analyzeGame);
chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (event) => { if (event.key === "Enter") sendChat(); });
undoButton.addEventListener("click", undoMove);
redoButton.addEventListener("click", redoMove);
document.querySelector("#backToGame").addEventListener("click", () => {
  document.querySelector("#reviewView").classList.remove("active");
  document.querySelector("#playView").classList.add("active");
});
document.querySelector("#flipBoard").addEventListener("click", () => {
  state.flipped = !state.flipped;
  renderBoard();
});
document.querySelector("#resignButton").addEventListener("click", () => {
  resignGame();
});
document.querySelectorAll(".preset-button").forEach((button) => {
  button.addEventListener("click", () => {
    state.difficulty = button.dataset.preset;
    const preset = presets[state.difficulty];
    depthInput.value = preset.depth;
    timeInput.value = preset.time;
    updateSettingLabels();
    document.querySelectorAll(".preset-button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });
});
function updateSettingLabels() {
  document.querySelector("#depthValue").textContent = depthInput.value;
  document.querySelector("#timeValue").textContent = `${Number(timeInput.value).toFixed(1)}s`;
}
depthInput.addEventListener("input", updateSettingLabels);
timeInput.addEventListener("input", updateSettingLabels);
uciInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendMove(uciInput.value.trim());
});

renderBoard();
renderAnalysis([]);
updateSettingLabels();
resignButton.disabled = true;
newGameButton.disabled = false;
undoButton.disabled = true;
redoButton.disabled = true;
resumeSavedGame();
setupGoogleLogin();
