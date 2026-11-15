// Bridge web UI: shows the live state from /events and sends card clicks to /pick.
(() => {
  const $ = (id) => document.getElementById(id);
  const LABELS = { starting: "starting", idle: "idle", listening: "listening", thinking: "thinking",
                   picking: "pick one", speaking: "speaking", offline: "disconnected" };
  const HINTS = {
    starting: "warming up…",
    idle: "hold space and speak",
    listening: "release space when done",
    thinking: "finding sentences…",
    picking: "press 1–3, thumbs up, or tap a card · 0 = none of these",
    speaking: "",
    offline: "reconnecting to bridge…",
  };
  let optionsKey = null;
  let current = null;

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function setStatus(status) {
    $("status").dataset.status = status;
    $("status-label").textContent = LABELS[status] || status;
    $("hint").textContent = HINTS[status] ?? "";
  }

  function renderHeard(fragments) {
    const box = $("heard");
    const key = JSON.stringify(fragments);
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    box.replaceChildren();
    const parts = fragments.map((f) => f.replace(/^[\s.,?!]+|[\s.,?!]+$/g, "")).filter(Boolean);
    if (!parts.length) { box.append(el("span", "empty", "–")); return; }
    parts.forEach((p, i) => {
      if (i) box.append(el("span", "sep", "…"));
      const s = el("span", "frag", p);
      s.style.animationDelay = `${i * 0.08}s`;
      box.append(s);
    });
  }

  function renderOptions(s) {
    const list = $("options");
    const key = JSON.stringify(s.options);
    if (key !== optionsKey) {          // new options: rebuild (cards fade in)
      optionsKey = key;
      list.replaceChildren();
      if (!s.options.length) {
        list.append(el("li", "empty-state", "say a few words and point at what you mean."));
      }
      s.options.forEach((text, i) => {
        const li = el("li", "card");
        li.tabIndex = 0;
        li.dataset.n = i + 1;
        li.append(el("span", "num", String(i + 1)), el("span", "text", text));
        li.addEventListener("click", () => pick(i + 1));
        li.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(i + 1); });
        list.append(li);
      });
    }
    const decided = s.chosen !== null && s.chosen !== undefined;
    list.classList.toggle("picking", s.status === "picking");
    list.classList.toggle("decided", decided);
    [...list.querySelectorAll(".card")].forEach((card, i) => {
      card.classList.toggle("chosen", i === s.chosen);
      card.classList.toggle("highlight", !decided && i === s.highlight);
    });
  }

  function renderTimings(t) {
    const parts = [["stt", t.stt], ["bonsai", t.bonsai], ["tts", t.tts]]
      .filter(([, v]) => v !== undefined).map(([k, v]) => `${k} ${v} ms`);
    $("timings").textContent = parts.join("  ·  ");
  }

  function render(s) {
    current = s;
    setStatus(s.status);

    $("camera-off").parentElement.classList.toggle("off", !s.camera);
    const pointing = $("pointing");
    pointing.classList.toggle("active", !!s.pointing_at);
    $("pointing-value").textContent = s.pointing_at || "nothing";
    $("pointing-method").textContent = s.pointing_at && s.pointing_method ? s.pointing_method : "";

    const thumbs = $("thumbs");
    thumbs.classList.toggle("show", !!s.thumbs);
    thumbs.classList.toggle("fired", !!(s.thumbs && s.thumbs.fired));
    if (s.thumbs) {
      $("thumbs-fill").style.width = `${Math.round(s.thumbs.progress * 100)}%`;
      $("thumbs-label").textContent = s.thumbs.fired ? "thumbs up ✓" : "thumbs up";
    }

    $("caregiver-field").classList.toggle("show", !!s.caregiver);
    $("caregiver").textContent = s.caregiver ? `“${s.caregiver}”` : "";

    renderHeard(s.fragments || []);
    renderOptions(s);
    renderTimings(s.timings_ms || {});

    $("urgent").classList.toggle("show", !!s.urgent);
    $("urgent-reason").textContent = s.urgent || "";
  }

  async function pick(n) {
    if (!current || current.status !== "picking") return;
    try {
      await fetch("/pick", { method: "POST", headers: { "Content-Type": "application/json" },
                             body: JSON.stringify({ n }) });
    } catch (e) { /* the next state update will show what happened */ }
  }

  function connect() {
    const source = new EventSource("/events");
    source.onmessage = (e) => render(JSON.parse(e.data));
    source.onerror = () => {
      setStatus("offline");
      // EventSource reconnects by itself; reload the video too once it's back.
    };
    source.onopen = () => {
      const video = $("video");
      video.src = `/video?t=${Date.now()}`;
    };
  }

  connect();
})();
