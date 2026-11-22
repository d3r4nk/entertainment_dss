const STORAGE_KEY = "entdec.chat-layout.v1";
const SCROLL_KEY = "entdec.chat-scroll.v1";
const COLUMN = '[data-testid="stColumn"]';
const FRAME = '[data-testid="stLayoutWrapper"]';
const MIN_WIDTH = 320;
const MIN_HEIGHT = 240;
const GAP = 18;

function readStorage(key) {
  try {
    return JSON.parse(sessionStorage.getItem(key)) || {};
  } catch {
    return {};
  }
}

function writeStorage(key, value) {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // The current layout still works when browser storage is disabled.
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function separator(parent, orientation, label, measure, change, signal) {
  const vertical = orientation === "vertical";
  const handle = document.createElement("div");
  handle.className = `entdec-resize-handle entdec-${vertical ? "split" : "height"}-handle`;
  handle.tabIndex = 0;
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", orientation);
  handle.setAttribute("aria-label", label);
  handle.title = label;

  let drag;
  const coordinate = (event) => vertical ? event.clientX : event.clientY;
  handle.addEventListener("pointerdown", (event) => {
    if (!event.isPrimary || event.button !== 0) return;
    drag = { id: event.pointerId, start: coordinate(event), value: measure() };
    handle.setPointerCapture(event.pointerId);
    handle.focus({ preventScroll: true });
    event.preventDefault();
  }, { signal });
  handle.addEventListener("pointermove", (event) => {
    if (drag?.id === event.pointerId) {
      change(drag.value + coordinate(event) - drag.start);
    }
  }, { signal });
  handle.addEventListener("lostpointercapture", () => { drag = undefined; }, { signal });
  handle.addEventListener("keydown", (event) => {
    const directions = vertical ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
    const direction = directions.indexOf(event.key);
    if (direction < 0) return;
    event.preventDefault();
    change(measure() + (direction === 0 ? -16 : 16));
  }, { signal });
  parent.append(handle);
  return handle;
}

function setRange(handle, value, min, max, suffix) {
  handle.setAttribute("aria-valuenow", String(Math.round(value)));
  handle.setAttribute("aria-valuemin", String(Math.round(min)));
  handle.setAttribute("aria-valuemax", String(Math.round(max)));
  handle.setAttribute("aria-valuetext", `${Math.round(value)}${suffix}`);
}

export default function ({ parentElement, data }) {
  const workspace = parentElement.closest(".st-key-entdec_workspace");
  if (!workspace) return;

  const saved = readStorage(STORAGE_KEY);
  const scroll = readStorage(SCROLL_KEY);
  const sizes = {
    ratio: Number.isFinite(saved.ratio) ? saved.ratio : .4,
    chat: Number.isFinite(saved.chat) ? saved.chat : data.height,
    trace: Number.isFinite(saved.trace) ? saved.trace : data.height,
  };
  let mounted;

  function attach(row, chat, trace) {
    const controller = new AbortController();
    const { signal } = controller;
    const handles = [];
    row.classList.add("entdec-pane-row");
    row.classList.toggle("entdec-chat-only", !trace);

    const availableWidth = () => row.clientWidth - GAP;
    const maxHeight = () => Math.max(MIN_HEIGHT, window.innerHeight - 120);
    const currentWidth = () => clamp(
      availableWidth() * sizes.ratio, MIN_WIDTH, availableWidth() - MIN_WIDTH,
    );
    const currentHeight = (name) => clamp(sizes[name], MIN_HEIGHT, maxHeight());

    let split;
    if (trace) {
      split = separator(row, "vertical", "Đổi chiều rộng khung chat và truy vấn", currentWidth,
        (value) => {
          const available = availableWidth();
          if (available < 2 * MIN_WIDTH) return;
          sizes.ratio = clamp(value, MIN_WIDTH, available - MIN_WIDTH) / available;
          updateSizes();
          writeStorage(STORAGE_KEY, sizes);
        }, signal);
      handles.push(split);
    }

    const panes = [];
    for (const [name, element, identity, label] of [
      ["chat", chat, data.conversation, "Đổi chiều cao khung chat"],
      ["trace", trace, data.trace, "Đổi chiều cao khung truy vấn"],
    ]) {
      if (!element) continue;
      const frame = element.parentElement;
      frame.classList.add("entdec-pane-frame");
      element.classList.add("entdec-pane-scroll");
      const handle = separator(frame, "horizontal", label, () => currentHeight(name),
        (value) => {
          sizes[name] = clamp(value, MIN_HEIGHT, maxHeight());
          updateSizes();
          writeStorage(STORAGE_KEY, sizes);
        }, signal);
      handles.push(handle);
      panes.push({ name, element, frame, handle });

      const previous = scroll[name]?.identity === identity ? scroll[name] : null;
      const position = { identity, top: previous?.top ?? 0, follow: previous?.follow ?? name === "chat" };
      scroll[name] = position;
      const restore = () => {
        element.scrollTop = position.follow ? element.scrollHeight : position.top;
      };
      panes.at(-1).restore = restore;
      element.addEventListener("scroll", () => {
        position.top = element.scrollTop;
        position.follow = name === "chat" &&
          element.scrollHeight - element.clientHeight - element.scrollTop <= 48;
        writeStorage(SCROLL_KEY, scroll);
      }, { passive: true, signal });
    }

    function updateSizes() {
      const sideBySide = Boolean(trace && availableWidth() >= 2 * MIN_WIDTH);
      row.classList.toggle("entdec-side-by-side", sideBySide);
      if (split) {
        split.hidden = !sideBySide;
        if (sideBySide) {
          row.style.setProperty("--entdec-chat-width", `${currentWidth()}px`);
          const percent = 100 / availableWidth();
          setRange(split, currentWidth() * percent, MIN_WIDTH * percent,
            (availableWidth() - MIN_WIDTH) * percent, "%");
        }
      }
      for (const { name, frame, handle } of panes) {
        const height = currentHeight(name);
        frame.style.setProperty("--entdec-pane-height", `${height}px`);
        setRange(handle, height, MIN_HEIGHT, maxHeight(), " px");
      }
    }

    updateSizes();
    for (const pane of panes) pane.restore();
    const rowObserver = new ResizeObserver(updateSizes);
    rowObserver.observe(row);
    window.addEventListener("resize", updateSizes, { signal });

    const chatContent = chat.querySelector(".st-key-entdec_chat_content");
    const contentObserver = new ResizeObserver(() => {
      if (scroll.chat?.follow) chat.scrollTop = chat.scrollHeight;
    });
    if (chatContent) contentObserver.observe(chatContent);

    return () => {
      controller.abort();
      rowObserver.disconnect();
      contentObserver.disconnect();
      for (const handle of handles) handle.remove();
      for (const { element, frame } of panes) {
        element.classList.remove("entdec-pane-scroll");
        frame.classList.remove("entdec-pane-frame");
        frame.style.removeProperty("--entdec-pane-height");
      }
      row.classList.remove("entdec-pane-row", "entdec-side-by-side", "entdec-chat-only");
      row.style.removeProperty("--entdec-chat-width");
    };
  }

  function connect() {
    const chat = workspace.querySelector(".st-key-entdec_chat_scroll");
    const trace = data.trace ? workspace.querySelector(".st-key-entdec_trace_scroll") : null;
    const row = chat?.closest(COLUMN)?.parentElement;
    if (
      !row || chat.parentElement?.matches(FRAME) !== true ||
      (data.trace && (!trace?.parentElement?.matches(FRAME) ||
        trace.closest(COLUMN)?.parentElement !== row))
    ) return;
    if (mounted?.chat === chat && mounted.trace === trace && mounted.row === row) return;
    mounted?.cleanup();
    mounted = { chat, trace, row, cleanup: attach(row, chat, trace) };
  }

  const observer = new MutationObserver(connect);
  observer.observe(workspace, { subtree: true, childList: true });
  connect();
  return () => {
    observer.disconnect();
    mounted?.cleanup();
  };
}
