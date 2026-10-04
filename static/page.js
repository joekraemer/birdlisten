// birdlisten collage page: click targets over the collage, the refresh
// swap, and the field-guide pop-up card. Plain ES2017, no libraries. Data
// is only ever inserted with textContent / setAttribute, never as HTML.
(function () {
  'use strict';

  const TICK_TIMEOUT_MS = 30000;
  let shown = JSON.parse(document.getElementById('layout').textContent);
  let c = document.getElementById('c');
  const targets = document.getElementById('targets');

  // ------------------------------------------------------------ targets
  function drawTargets(layout) {
    const active = document.activeElement;
    const focusedSci = active && active.classList && active.classList.contains('t')
      ? active.getAttribute('data-sci') : null;
    while (targets.firstChild) targets.removeChild(targets.firstChild);
    let refocus = null;
    layout.targets.forEach(function (t) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 't';
      b.setAttribute('aria-label', t.common_name);
      b.setAttribute('data-sci', t.scientific_name);
      b.style.left = t.x + '%';
      b.style.top = t.y + '%';
      b.style.width = t.w + '%';
      b.style.height = t.h + '%';
      b.addEventListener('click', function () { openCard(t.scientific_name, t, b); });
      targets.appendChild(b);
      if (focusedSci !== null && t.scientific_name === focusedSci) refocus = b;
    });
    if (focusedSci !== null) (refocus || c).focus();
  }

  function targetFor(layout, sci) {
    for (let i = 0; i < layout.targets.length; i++) {
      if (layout.targets[i].scientific_name === sci) return layout.targets[i];
    }
    return null;
  }

  // ------------------------------------------------------------ refresh
  let inflight = false;
  async function tick() {
    if (inflight) return;
    inflight = true;
    const ac = new AbortController();
    let timer;
    const deadline = new Promise(function (_, reject) {
      timer = setTimeout(function () { ac.abort(); reject(new Error('tick timeout')); }, TICK_TIMEOUT_MS);
    });
    deadline.catch(function () {});   // only ever awaited inside Promise.race
    try {
      const url = '/api/layout?hours=' + shown.hours + '&w=' + shown.w + '&h=' + shown.h;
      const r = await Promise.race([fetch(url, {cache: 'no-store', signal: ac.signal}), deadline]);
      if (!r.ok) return;
      const layout = await Promise.race([r.json(), deadline]);
      if (layout.token === shown.token) return;
      const im = new Image();
      im.id = 'c'; im.alt = c.alt; im.tabIndex = -1;
      im.width = layout.w; im.height = layout.h;
      im.src = layout.png;                   // /collage.png?v=<token>, immutable
      try {
        await Promise.race([im.decode(), deadline]);
      } catch (e) {
        im.src = '';                         // stop a hung image download
        throw e;
      }
      if (layout.token === shown.token) return;
      const hadFocus = document.activeElement === c;
      c.replaceWith(im); c = im;
      drawTargets(layout); shown = layout;
      if (hadFocus) c.focus();
    } catch (e) {
      // any failure: keep the old image and old targets; the next tick retries
    } finally {
      clearTimeout(timer);
      inflight = false;
    }
  }

  // ------------------------------------------------------------ card
  function openCard(sci, target, opener) {
    // filled in by the pop-up card code below
  }

  drawTargets(shown);
  setInterval(tick, Number(document.body.dataset.refreshMs) || 60000);
})();
