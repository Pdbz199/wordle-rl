const STORAGE_KEY = "wordle_rl_turns_v1";
const DEFAULT_PATTERN = ["b", "b", "b", "b", "b"];
const COLOR_ORDER = { b: 1, y: 2, g: 3 };

const state = {
  turns: [],
  draftGuess: "",
  draftPattern: [...DEFAULT_PATTERN],
  lastResponse: null,
  authenticated: false,
};

const loginCard = document.getElementById("login-card");
const assistantCard = document.getElementById("assistant-card");
const messageEl = document.getElementById("message");
const nextGuessEl = document.getElementById("next-guess");
const candidateCountEl = document.getElementById("candidate-count");
const turnCountEl = document.getElementById("turn-count");
const boardEl = document.getElementById("board");
const topChoicesCards = document.getElementById("top-choices-cards");
const patternFallbackEl = document.getElementById("pattern-fallback");
const topbarActions = document.getElementById("topbar-actions");

const loginForm = document.getElementById("login-form");
const logoutBtn = document.getElementById("logout-btn");
const undoBtn = document.getElementById("undo-btn");
const resetBtn = document.getElementById("reset-btn");
const useSuggestionBtn = document.getElementById("use-suggestion-btn");
const submitTurnBtn = document.getElementById("submit-turn-btn");
const keyButtons = Array.from(document.querySelectorAll(".key-btn"));
const infoDots = Array.from(document.querySelectorAll(".info-dot"));

function setMessage(text, isError = false) {
  messageEl.textContent = text || "";
  messageEl.style.color = isError ? "#b42318" : "#19673f";
}

function normalizeGuess(value) {
  return (value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z]/g, "")
    .slice(0, 5);
}

function normalizePatternInput(value) {
  return (value || "")
    .trim()
    .toLowerCase()
    .replaceAll("x", "b")
    .replace(/[^byg]/g, "")
    .slice(0, 5);
}

function normalizePattern(value) {
  const cleaned = normalizePatternInput(value);
  if (cleaned.length === 5) {
    return cleaned;
  }
  return "bbbbb";
}

function patternArrayFromText(value) {
  return normalizePattern(value).split("");
}

function validateGuess(guess) {
  return /^[a-z]{5}$/.test(guess);
}

function validatePattern(pattern) {
  return /^[byg]{5}$/.test(pattern);
}

function isSolved() {
  if (state.lastResponse?.is_solved) {
    return true;
  }
  const lastTurn = state.turns[state.turns.length - 1];
  return Boolean(lastTurn && lastTurn.pattern === "ggggg");
}

function canEditDraft() {
  return state.authenticated && state.turns.length < 6 && !isSolved();
}

function normalizeTurns(rawTurns) {
  if (!Array.isArray(rawTurns)) {
    return [];
  }
  return rawTurns
    .map((turn) => ({
      guess: normalizeGuess(turn?.guess || ""),
      pattern: normalizePattern(turn?.pattern || ""),
    }))
    .filter((turn) => validateGuess(turn.guess) && validatePattern(turn.pattern))
    .slice(0, 6);
}

function loadState() {
  state.turns = [];
  state.draftGuess = "";
  state.draftPattern = [...DEFAULT_PATTERN];

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }

    const parsed = JSON.parse(raw);

    // Backward compatibility with previous storage format: array of turns.
    if (Array.isArray(parsed)) {
      state.turns = normalizeTurns(parsed);
      return;
    }

    if (!parsed || typeof parsed !== "object") {
      return;
    }

    state.turns = normalizeTurns(parsed.turns);
    state.draftGuess = normalizeGuess(parsed.draftGuess || "");
    state.draftPattern = patternArrayFromText(parsed.draftPattern || "bbbbb");
  } catch (_err) {
    state.turns = [];
    state.draftGuess = "";
    state.draftPattern = [...DEFAULT_PATTERN];
  }
}

function persistState() {
  const payload = {
    version: 2,
    turns: state.turns,
    draftGuess: state.draftGuess,
    draftPattern: state.draftPattern.join(""),
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function tileColorClass(symbol) {
  if (symbol === "g") {
    return "tile--correct";
  }
  if (symbol === "y") {
    return "tile--present";
  }
  return "tile--absent";
}

function createTile(letter = "", classNames = []) {
  const tile = document.createElement("button");
  tile.type = "button";
  tile.className = ["tile", ...classNames].join(" ");
  tile.textContent = letter;
  return tile;
}

function cycleDraftPattern(index) {
  if (!canEditDraft()) {
    return;
  }
  if (state.draftGuess.length !== 5) {
    setMessage("Type a 5-letter guess first, then set tile colors.", true);
    return;
  }
  const current = state.draftPattern[index];
  const next = current === "b" ? "y" : current === "y" ? "g" : "b";
  state.draftPattern[index] = next;
  persistState();
  renderBoard();
  syncPatternFallback();
}

function renderBoard() {
  boardEl.innerHTML = "";
  const activeRow = canEditDraft() ? state.turns.length : -1;
  const suggestion = (state.lastResponse?.next_guess || "").toLowerCase();

  for (let rowIndex = 0; rowIndex < 6; rowIndex += 1) {
    const row = document.createElement("div");
    row.className = "board-row";

    if (rowIndex < state.turns.length) {
      const turn = state.turns[rowIndex];
      for (let colIndex = 0; colIndex < 5; colIndex += 1) {
        const tile = createTile(turn.guess[colIndex].toUpperCase(), [tileColorClass(turn.pattern[colIndex])]);
        row.appendChild(tile);
      }
      boardEl.appendChild(row);
      continue;
    }

    if (rowIndex === activeRow) {
      for (let colIndex = 0; colIndex < 5; colIndex += 1) {
        const typedLetter = state.draftGuess[colIndex] || "";
        const ghostLetter = typedLetter ? "" : suggestion[colIndex] || "";
        const letter = (typedLetter || ghostLetter).toUpperCase();
        const classes = ["tile--active"];

        if (typedLetter) {
          if (state.draftGuess.length === 5) {
            classes.push(tileColorClass(state.draftPattern[colIndex]));
          }
        } else if (ghostLetter) {
          classes.push("tile--ghost");
        }

        const tile = createTile(letter, classes);
        tile.addEventListener("click", () => cycleDraftPattern(colIndex));
        row.appendChild(tile);
      }
      boardEl.appendChild(row);
      continue;
    }

    for (let colIndex = 0; colIndex < 5; colIndex += 1) {
      row.appendChild(createTile("", []));
    }
    boardEl.appendChild(row);
  }
}

function renderStats() {
  turnCountEl.textContent = String(state.turns.length);

  if (!state.lastResponse) {
    nextGuessEl.textContent = "-";
    candidateCountEl.textContent = "-";
    return;
  }

  nextGuessEl.textContent = state.lastResponse.next_guess
    ? state.lastResponse.next_guess.toUpperCase()
    : state.lastResponse.is_solved
      ? "SOLVED"
      : "-";
  candidateCountEl.textContent = String(state.lastResponse.candidate_count ?? "-");
}

function renderTopChoices() {
  topChoicesCards.innerHTML = "";
  const choices = state.lastResponse?.top_choices || [];

  if (!choices.length) {
    const empty = document.createElement("p");
    empty.className = "choice-empty";
    empty.textContent = "No alternatives available for this state.";
    topChoicesCards.appendChild(empty);
    return;
  }

  const topProbability = Math.max(Number(choices[0]?.probability || 0), 0.000001);

  choices.forEach((choice, index) => {
    const choiceWord = normalizeGuess(choice.word || "");
    if (!choiceWord) {
      return;
    }
    const probability = Number(choice.probability || 0);
    const percent = probability * 100;
    const barPct = Math.max(3, Math.min(100, (probability / topProbability) * 100));

    const card = document.createElement("article");
    card.className = "choice-card";
    card.innerHTML = `
      <div class="choice-head">
        <span class="choice-rank">#${index + 1}</span>
        <span class="choice-word">${choiceWord.toUpperCase()}</span>
        <span class="choice-prob">${percent.toFixed(3)}%</span>
      </div>
      <div class="choice-bar"><div class="choice-bar-fill" style="width: ${barPct}%"></div></div>
      <button class="choice-action" type="button" data-word="${choiceWord}">Use this</button>
    `;
    topChoicesCards.appendChild(card);
  });
}

function renderKeyboard() {
  const bestStatus = {};

  state.turns.forEach((turn) => {
    for (let index = 0; index < 5; index += 1) {
      const letter = turn.guess[index];
      const symbol = turn.pattern[index];
      if (!letter || !symbol) {
        continue;
      }
      const previous = bestStatus[letter] || 0;
      const current = COLOR_ORDER[symbol] || 0;
      if (current > previous) {
        bestStatus[letter] = current;
      }
    }
  });

  keyButtons.forEach((button) => {
    const key = button.dataset.key || "";
    button.classList.remove("key--absent", "key--present", "key--correct");
    if (!/^[a-z]$/.test(key)) {
      return;
    }
    const rank = bestStatus[key];
    if (rank === COLOR_ORDER.g) {
      button.classList.add("key--correct");
    } else if (rank === COLOR_ORDER.y) {
      button.classList.add("key--present");
    } else if (rank === COLOR_ORDER.b) {
      button.classList.add("key--absent");
    }
  });
}

function syncPatternFallback() {
  const enabled = canEditDraft() && state.draftGuess.length === 5;
  patternFallbackEl.disabled = !enabled;
  if (document.activeElement !== patternFallbackEl) {
    patternFallbackEl.value = state.draftPattern.join("");
  }
}

function renderAll() {
  renderBoard();
  renderStats();
  renderTopChoices();
  renderKeyboard();
  syncPatternFallback();
}

function showLoggedInUi() {
  loginCard.classList.add("hidden");
  assistantCard.classList.remove("hidden");
  topbarActions.classList.remove("hidden");
}

function showLoggedOutUi() {
  assistantCard.classList.add("hidden");
  loginCard.classList.remove("hidden");
  topbarActions.classList.add("hidden");
}

function closeAllTooltips(except = null) {
  infoDots.forEach((dot) => {
    if (dot === except) {
      return;
    }
    dot.classList.remove("tooltip-open");
    dot.setAttribute("aria-expanded", "false");
  });
}

function wireTooltips() {
  infoDots.forEach((dot) => {
    dot.setAttribute("aria-expanded", "false");
    dot.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const shouldOpen = !dot.classList.contains("tooltip-open");
      closeAllTooltips();
      if (shouldOpen) {
        dot.classList.add("tooltip-open");
        dot.setAttribute("aria-expanded", "true");
      }
    });
  });

  document.addEventListener("click", () => {
    closeAllTooltips();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeAllTooltips();
    }
  });
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
    credentials: "include",
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch (_err) {
    payload = null;
  }

  if (!response.ok) {
    const message = payload?.detail || `Request failed with status ${response.status}.`;
    throw new Error(message);
  }
  return payload;
}

async function refreshSuggestion() {
  if (!state.authenticated) {
    return;
  }

  try {
    const response = await apiRequest("/api/suggest", {
      method: "POST",
      body: {
        turns: state.turns,
        top_k: 10,
      },
    });
    state.lastResponse = response;
    renderAll();
    if (response.is_solved) {
      setMessage("Puzzle solved. Start a new puzzle when ready.");
    } else {
      setMessage("");
    }
  } catch (error) {
    state.lastResponse = null;
    renderAll();
    setMessage(error.message, true);
  }
}

async function checkAuth() {
  try {
    const status = await apiRequest("/auth/status");
    state.authenticated = Boolean(status.authenticated);
  } catch (_err) {
    state.authenticated = false;
  }

  if (state.authenticated) {
    showLoggedInUi();
    await refreshSuggestion();
  } else {
    showLoggedOutUi();
    renderAll();
  }
}

async function login(event) {
  event.preventDefault();
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;

  try {
    await apiRequest("/auth/login", {
      method: "POST",
      body: { username, password },
    });
    state.authenticated = true;
    showLoggedInUi();
    setMessage("Signed in.");
    await refreshSuggestion();
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function logout() {
  try {
    await apiRequest("/auth/logout", { method: "POST" });
  } catch (_err) {
    // Keep local logout behavior even when the logout request fails.
  }
  state.authenticated = false;
  state.lastResponse = null;
  showLoggedOutUi();
  renderAll();
  setMessage("Signed out.");
}

async function submitTurn() {
  if (!canEditDraft()) {
    return;
  }

  const guess = state.draftGuess;
  const pattern = state.draftPattern.join("");

  if (!validateGuess(guess)) {
    setMessage("Guess must be exactly 5 letters.", true);
    return;
  }
  if (!validatePattern(pattern)) {
    setMessage("Pattern must be exactly 5 chars using b/y/g.", true);
    return;
  }

  state.turns.push({ guess, pattern });
  state.draftGuess = "";
  state.draftPattern = [...DEFAULT_PATTERN];
  persistState();
  await refreshSuggestion();
}

async function undoLastTurn() {
  if (!state.turns.length) {
    setMessage("There is no turn to undo.", true);
    return;
  }
  const removed = state.turns.pop();
  state.draftGuess = removed.guess;
  state.draftPattern = patternArrayFromText(removed.pattern);
  persistState();
  await refreshSuggestion();
}

async function resetPuzzle() {
  state.turns = [];
  state.draftGuess = "";
  state.draftPattern = [...DEFAULT_PATTERN];
  state.lastResponse = null;
  persistState();
  setMessage("Started a new puzzle.");
  await refreshSuggestion();
}

function applySuggestionWord(word) {
  if (!canEditDraft()) {
    return;
  }
  const suggestion = normalizeGuess(word);
  if (!validateGuess(suggestion)) {
    setMessage("No active suggestion to apply.", true);
    return;
  }
  state.draftGuess = suggestion;
  persistState();
  renderAll();
  setMessage("Suggestion loaded into the active row.");
}

function handleVirtualKey(rawKey) {
  const key = rawKey.toLowerCase();

  if (key === "enter") {
    void submitTurn();
    return;
  }

  if (!canEditDraft()) {
    return;
  }

  if (key === "backspace") {
    state.draftGuess = state.draftGuess.slice(0, -1);
    persistState();
    renderAll();
    return;
  }

  if (!/^[a-z]$/.test(key)) {
    return;
  }

  if (state.draftGuess.length >= 5) {
    return;
  }

  state.draftGuess += key;
  persistState();
  renderAll();
}

function onGlobalKeyDown(event) {
  if (!state.authenticated || assistantCard.classList.contains("hidden")) {
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }

  const target = event.target;
  const tag = (target?.tagName || "").toLowerCase();
  const id = target?.id || "";

  if (tag === "input" && id === "pattern-fallback") {
    if (event.key === "Enter") {
      event.preventDefault();
      void submitTurn();
    }
    return;
  }

  if (tag === "input" || tag === "textarea") {
    return;
  }

  const key = event.key.toLowerCase();
  if (key === "enter" || key === "backspace" || /^[a-z]$/.test(key)) {
    event.preventDefault();
    handleVirtualKey(key);
  }
}

function onPatternFallbackInput() {
  const cleaned = normalizePatternInput(patternFallbackEl.value);
  patternFallbackEl.value = cleaned;
  if (cleaned.length === 5) {
    state.draftPattern = cleaned.split("");
    persistState();
    renderBoard();
    setMessage("");
  }
}

function onPatternFallbackBlur() {
  patternFallbackEl.value = state.draftPattern.join("");
}

function wireEvents() {
  loginForm.addEventListener("submit", login);
  logoutBtn.addEventListener("click", logout);
  submitTurnBtn.addEventListener("click", () => {
    void submitTurn();
  });
  undoBtn.addEventListener("click", () => {
    void undoLastTurn();
  });
  resetBtn.addEventListener("click", () => {
    void resetPuzzle();
  });
  useSuggestionBtn.addEventListener("click", () => {
    applySuggestionWord(state.lastResponse?.next_guess || "");
  });

  patternFallbackEl.addEventListener("input", onPatternFallbackInput);
  patternFallbackEl.addEventListener("blur", onPatternFallbackBlur);

  keyButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.key || "";
      handleVirtualKey(key);
    });
  });

  topChoicesCards.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-word]");
    if (!button) {
      return;
    }
    applySuggestionWord(button.dataset.word || "");
  });

  document.addEventListener("keydown", onGlobalKeyDown);
  wireTooltips();
}

loadState();
persistState();
renderAll();
wireEvents();
void checkAuth();
