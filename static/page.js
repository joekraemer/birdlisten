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
  const SVG = 'http://www.w3.org/2000/svg';
  const LINK_HOSTS = ['en.wikipedia.org', 'ebird.org', 'www.allaboutbirds.org',
                      'www.wikidata.org', 'creativecommons.org'];
  const card = document.getElementById('card');
  const scrim = document.getElementById('scrim');
  const titleEl = document.getElementById('card-title');
  const sciEl = card.querySelector('.sci');
  const plateEl = card.querySelector('.plate');
  const closeBtn = card.querySelector('.close');
  const bodyEl = card.querySelector('.body');
  const heardEl = card.querySelector('.heard .content');
  const aboutSec = card.querySelector('section.about');
  const aboutEl = aboutSec.querySelector('.content');
  let reqId = 0;
  let openSci = null;
  let followTimer = null;
  let lastData = null;                      // the newest /api/species response for the open card

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function svg(tag, attrs, text) {
    const e = document.createElementNS(SVG, tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, String(attrs[k])); });
    if (text) e.textContent = text;
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function safeLink(url, text) {
    let u;
    try { u = new URL(url); } catch (e) { return null; }
    if (u.protocol !== 'https:' || LINK_HOSTS.indexOf(u.hostname) < 0) return null;
    const a = el('a', null, text);
    a.setAttribute('href', u.href);
    a.setAttribute('target', '_blank');
    a.setAttribute('rel', 'noopener noreferrer');
    return a;
  }

  function placeholder() {
    clear(plateEl);
    plateEl.appendChild(el('div', 'plate-ph'));
  }
  function setPlate(url) {
    if (typeof url !== 'string' || !/^\/plate\/[a-z0-9-]{1,100}\.png$/.test(url)) { placeholder(); return; }
    const cur = plateEl.querySelector('img');
    if (cur && cur.getAttribute('src') === url) return;
    clear(plateEl);
    const img = el('img');
    img.setAttribute('alt', '');
    img.addEventListener('error', placeholder);
    img.setAttribute('src', url);
    plateEl.appendChild(img);
  }

  function windowPhrase(hours) {
    if (hours === 1) return 'the last hour';
    if (hours % 24 === 0 && hours > 24) return 'the last ' + (hours / 24) + ' days';
    return 'the last ' + hours + ' hours';
  }
  function pct(v) { return Math.round(v * 100) + '%'; }
  function h12(h) { return (h % 12) || 12; }
  function ampm(h) { return h < 12 ? 'AM' : 'PM'; }
  function hourSpan(h) {
    const n = (h + 1) % 24;
    return ampm(h) === ampm(n)
      ? h12(h) + '\u2013' + h12(n) + ' ' + ampm(h)
      : h12(h) + ' ' + ampm(h) + '\u2013' + h12(n) + ' ' + ampm(n);
  }

  // Detections per local hour: 24 thin bars, field-guide style. With a
  // typical day (#13), its bars are drawn wide and light and the window's
  // narrow and dark on top. Each is scaled to its own peak, so the light bars
  // show the shape of a usual day and the dark ones when it was heard.
  function chart(byHour, busiest, hours, typical) {
    let label = byHour.some(Boolean) ? 'Most detections ' + hourSpan(busiest) : 'Not heard in ' + windowPhrase(hours);
    if (typical) label += '; usually busiest ' + hourSpan(typical.indexOf(Math.max.apply(null, typical)));
    const s = svg('svg', {viewBox: '0 0 240 64', class: 'chart', role: 'img', 'aria-label': label});
    const bars = function (vals, x, w, fill, cls) {
      const max = Math.max.apply(null, vals);
      vals.forEach(function (v, i) {
        if (!v) return;
        const h = Math.max(1, Math.round(44 * v / max));
        s.appendChild(svg('rect', {x: 10 * i + x, y: 48 - h, width: w, height: h, fill: fill, class: cls}));
      });
    };
    if (typical) {
      bars(typical, 1, 8, TYPICAL_FILL, 'typ');
      bars(byHour, 3, 4, HEARD_FILL, 'now');
    } else {
      bars(byHour, 2, 6, HEARD_FILL, 'now');
    }
    s.appendChild(svg('rect', {x: 0, y: 48, width: 240, height: 1, fill: '#baaa8e'}));
    [[0, '12a'], [6, '6a'], [12, '12p'], [18, '6p']].forEach(function (t) {
      s.appendChild(svg('rect', {x: 10 * t[0] + 4.5, y: 49, width: 1, height: 3, fill: '#baaa8e'}));
      s.appendChild(svg('text', {x: 10 * t[0] + 2, y: 60, fill: '#7c705e'}, t[1]));
    });
    return s;
  }
  const HEARD_FILL = '#7c705e';
  const TYPICAL_FILL = '#d8cab0';

  function validTypical(h) {
    const t = h.typical_by_hour;
    return Array.isArray(t) && t.length === 24 && t.every(function (v) { return typeof v === 'number' && v >= 0; })
      && t.some(Boolean) ? t : null;
  }

  function fillHeard(data) {
    clear(heardEl);
    const h = data.heard || {};
    const typical = validTypical(h);
    const caption = typical
      ? 'Dark: ' + windowPhrase(data.hours) + '. Light: when it is usually heard, over the last ' +
        h.typical_days + ' days. Local time.'
      : 'Detections by hour of day, local time';
    if (!h.count) {
      heardEl.appendChild(el('p', 'quiet', 'Not heard in ' + windowPhrase(data.hours)));
      if (typical) {
        heardEl.appendChild(chart(h.by_hour || new Array(24).fill(0), 0, data.hours, typical));
        heardEl.appendChild(el('p', 'credit', caption));
      }
      return;
    }
    heardEl.appendChild(el('p', null, h.count + (h.count === 1 ? ' detection' : ' detections') +
      ' \u00b7 best ' + pct(h.max_conf) + ', typical ' + pct(h.median_conf)));
    heardEl.appendChild(el('p', null, h.count === 1 ? 'Heard ' + h.last_local
      : 'First ' + h.first_local + ' \u00b7 last ' + h.last_local));
    if (h.cameras && h.cameras.length) {
      heardEl.appendChild(el('p', null, 'Cameras: ' + h.cameras.join(', ')));
    }
    heardEl.appendChild(chart(h.by_hour, h.busiest_hour, data.hours, typical));
    heardEl.appendChild(el('p', 'credit', caption));
  }

  function credit(prefix, parts) {
    const p = el('p', 'credit', prefix);
    parts.forEach(function (part) {
      if (typeof part === 'string') { p.appendChild(document.createTextNode(part)); return; }
      const a = safeLink(part[0], part[1]);
      p.appendChild(a || document.createTextNode(part[1]));
    });
    return p;
  }

  // final: the last pass for this card (the follow-up). Nothing else is
  // fetched after it, so it must not promise more notes (#6).
  function fillAbout(data, final) {
    if (!data.binomial) { aboutSec.hidden = true; return; }
    aboutSec.hidden = false;
    const keepScroll = bodyEl.scrollTop;
    const active = document.activeElement;
    const activeHref = active && aboutEl.contains(active) ? active.getAttribute('href') : null;
    clear(aboutEl);
    const f = data.facts || {};
    const w = f.wikipedia;
    let any = false;
    if (w && typeof w.extract === 'string' && safeLink(w.url, w.title)) {
      aboutEl.appendChild(el('p', 'blurb', w.extract));
      aboutEl.appendChild(credit(w.trimmed ? 'Excerpt from Wikipedia: ' : 'From Wikipedia: ',
        [[w.url, w.title], ', ', [w.license_url, w.license]]));
      any = true;
    }
    const sz = f.size;
    if (sz && (sz.mass || sz.length || sz.wingspan)) {
      const dl = el('dl', 'size');
      [['Mass', sz.mass], ['Length', sz.length], ['Wingspan', sz.wingspan]].forEach(function (row) {
        if (!row[1]) return;
        dl.appendChild(el('dt', null, row[0]));
        dl.appendChild(el('dd', null, row[1]));
      });
      aboutEl.appendChild(dl);
      // One credit per source; with two, say which fields each supplied.
      const parts = [];
      (sz.sources || []).forEach(function (s, i) {
        if (i) parts.push('; ');
        parts.push([s.url, s.name], ' (', [s.license_url, s.license], ')');
        if (sz.sources.length > 1 && s.fields) parts.push(' for ' + s.fields.join(', '));
      });
      aboutEl.appendChild(credit('Size: ', parts));
      any = true;
    }
    const nb = f.nearby;
    if (nb) {
      let text;
      if (nb.reported) {
        const d = nb.days_ago;
        text = 'Reported on eBird within ' + nb.dist_km + ' km ' +
          (d === 0 ? 'today' : d === 1 ? 'yesterday' : d + ' days ago');
      } else {
        text = 'No eBird reports within ' + nb.dist_km + ' km in the last ' + nb.back_days + ' days';
      }
      aboutEl.appendChild(el('p', 'nearby', text));
      aboutEl.appendChild(credit('Data from ', [[nb.url, 'eBird.org'], ' (Cornell Lab of Ornithology)']));
      any = true;
    }
    const links = el('p', 'links');
    [[f.ebird && f.ebird.url, 'eBird'], [data.links && data.links.allaboutbirds, 'All About Birds'],
     [w && w.url, 'Wikipedia']].forEach(function (l) {
      if (!l[0]) return;
      const a = safeLink(l[0], l[1]);
      if (!a) return;
      if (links.firstChild) links.appendChild(document.createTextNode(' \u00b7 '));
      links.appendChild(a);
    });
    if (!final && data.pending && data.pending.length) {
      aboutEl.appendChild(el('p', 'loading', any ? 'Gathering more notes\u2026' : 'Gathering notes\u2026'));
    } else if (!any) {
      aboutEl.appendChild(el('p', 'quiet', 'No notes for this bird yet.'));
    }
    if (links.firstChild) aboutEl.appendChild(links);
    bodyEl.scrollTop = keepScroll;
    if (activeHref) {
      const again = Array.prototype.find.call(aboutEl.querySelectorAll('a'),
        function (a) { return a.getAttribute('href') === activeHref; });
      (again || closeBtn).focus();
    }
  }

  function failed() {
    clear(heardEl);
    heardEl.appendChild(el('p', 'quiet', 'Couldn\u2019t load details'));
    aboutSec.hidden = true;
  }

  async function load(sci, my, aboutOnly) {
    let data;
    try {
      const r = await fetch('/api/species/' + encodeURIComponent(sci) + '?hours=' + shown.hours,
                            {cache: 'no-store'});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      data = await r.json();
    } catch (e) {
      if (my !== reqId) return;
      if (!aboutOnly) failed();
      else if (lastData) fillAbout(lastData, true);   // follow-up failed: keep what we had, no spinner
      return;
    }
    if (my !== reqId) return;               // the card was closed or replaced
    lastData = data;
    if (aboutOnly) { fillAbout(data, true); return; }
    titleEl.textContent = data.common_name;
    setPlate(data.plate_url);
    fillHeard(data);
    fillAbout(data, false);
    if (data.pending && data.pending.length) {
      followTimer = setTimeout(function () { load(sci, my, true); }, 4000);
    }
  }

  function openCard(sci, target) {
    reqId += 1;
    clearTimeout(followTimer);
    lastData = null;
    openSci = sci;
    titleEl.textContent = target ? target.common_name : sci;
    sciEl.textContent = sci;
    if (target && target.art && target.stem) setPlate('/plate/' + target.stem + '.png');
    else placeholder();
    clear(heardEl);
    heardEl.appendChild(el('p', 'loading', 'Gathering notes\u2026'));
    clear(aboutEl);
    aboutSec.hidden = false;
    card.hidden = false;
    scrim.hidden = false;
    document.documentElement.classList.add('modal');
    bodyEl.scrollTop = 0;
    closeBtn.focus();
    load(sci, reqId, false);
  }

  function closeCard() {
    if (card.hidden) return;
    reqId += 1;
    clearTimeout(followTimer);
    card.hidden = true;
    scrim.hidden = true;
    document.documentElement.classList.remove('modal');
    if (location.hash.indexOf('#species=') === 0 && history.replaceState) {
      history.replaceState(null, '', location.pathname + location.search);
    }
    const back = Array.prototype.find.call(targets.children,
      function (b) { return b.getAttribute('data-sci') === openSci; });
    (back || c).focus();
    openSci = null;
  }

  function focusables() {
    return Array.prototype.filter.call(card.querySelectorAll('button, a[href]'),
      function (e) { return e.offsetParent !== null || e === closeBtn; });
  }

  document.addEventListener('keydown', function (e) {
    if (card.hidden) return;
    if (e.key === 'Escape' || e.key === 'Esc') { e.preventDefault(); closeCard(); return; }
    if (e.key !== 'Tab') return;
    const f = focusables();
    if (!f.length) return;
    const first = f[0];
    const last = f[f.length - 1];
    if (!card.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
    else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
  closeBtn.addEventListener('click', closeCard);
  scrim.addEventListener('click', closeCard);

  // /#species=<scientific name> opens that card (used for headless screenshots).
  function fromHash() {
    const h = location.hash;
    if (h.indexOf('#species=') !== 0) return;
    let name;
    try { name = decodeURIComponent(h.slice('#species='.length)); } catch (e) { return; }
    if (!(name.length > 0 && name.length <= 100)) return;
    openCard(name, targetFor(shown, name));
  }

  drawTargets(shown);
  setInterval(tick, Number(document.body.dataset.refreshMs) || 60000);
  window.addEventListener('hashchange', fromHash);
  fromHash();
})();
