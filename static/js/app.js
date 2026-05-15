/* ============================================================
   NBA Predictor — frontend interactions
   ============================================================ */

(function () {
  'use strict';

  // ── Toasts ────────────────────────────────────────────────
  function toast(msg, kind) {
    const wrap = document.getElementById('toast-wrap');
    if (!wrap) return;
    const t = document.createElement('div');
    t.className = 'toast ' + (kind || '');
    t.textContent = msg;
    wrap.appendChild(t);
    setTimeout(() => {
      t.style.transition = 'opacity 0.25s, transform 0.25s';
      t.style.opacity = '0';
      t.style.transform = 'translateY(8px)';
      setTimeout(() => t.remove(), 280);
    }, 3000);
  }

  // ── Backend toggle ────────────────────────────────────────
  async function toggleBackend() {
    if (!window.TRANSFORMER_AVAILABLE) {
      toast('Transformer artifacts not yet built — run train_transformer.py first', 'info');
      return;
    }
    const next = window.BACKEND === 'transformer' ? 'gbm' : 'transformer';
    try {
      const resp = await fetch('/api/backend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: next }),
      });
      if (resp.ok) {
        toast('Switched to ' + (next === 'transformer' ? 'Transformer (v3 PyTorch)' : 'GBM (v2 XGBoost)'),
              next === 'transformer' ? 'violet' : 'info');
        setTimeout(() => location.reload(), 380);
      } else {
        toast('Failed to switch backend', 'info');
      }
    } catch (e) {
      toast('Network error switching backend', 'info');
    }
  }

  // ── Command palette ───────────────────────────────────────
  const cmdkState = { focused: 0, items: [] };

  function buildCmdItems(query) {
    const q = (query || '').toLowerCase().trim();
    const items = [];

    // Navigation commands
    const nav = [
      { kind: 'page', icon: '⌂', label: 'Dashboard',     meta: 'home',     href: '/' },
      { kind: 'page', icon: '▤', label: 'Upcoming Games', meta: 'live',     href: '/upcoming' },
      { kind: 'page', icon: '⊕', label: 'Manual Predict', meta: 'matchup',  href: '/predict' },
      { kind: 'page', icon: '⚛', label: 'Models & Metrics', meta: 'insights', href: '/models' },
    ];
    nav.forEach(n => {
      if (!q || n.label.toLowerCase().includes(q)) items.push(n);
    });

    // Backend switch
    if (window.TRANSFORMER_AVAILABLE && (!q || 'switch backend transformer gbm'.includes(q))) {
      items.push({
        kind: 'action',
        icon: window.BACKEND === 'transformer' ? '⚡' : '🧠',
        label: 'Switch to ' + (window.BACKEND === 'transformer' ? 'GBM' : 'Transformer'),
        meta: 'backend',
        run: toggleBackend,
      });
    }

    // Teams
    const team_info = window.TEAM_INFO || {};
    const matches = [];
    for (const [abbr, info] of Object.entries(team_info)) {
      const hay = (info.name + ' ' + abbr).toLowerCase();
      if (!q || hay.includes(q)) {
        matches.push({
          kind: 'team',
          icon_img: info.logo,
          abbr,
          label: info.name,
          meta: abbr,
          href: '/team/' + abbr,
        });
      }
    }
    matches.sort((a, b) => a.label.localeCompare(b.label));
    matches.slice(0, 12).forEach(m => items.push(m));

    return items;
  }

  function renderCmdK(query) {
    const list = document.getElementById('cmdk-list');
    const items = buildCmdItems(query);
    cmdkState.items = items;
    cmdkState.focused = 0;

    if (!items.length) {
      list.innerHTML = '<div class="cmdk-empty">No results</div>';
      return;
    }

    // group by kind
    const groups = { page: 'Pages', action: 'Actions', team: 'Teams' };
    const byKind = {};
    items.forEach(it => { (byKind[it.kind] = byKind[it.kind] || []).push(it); });

    let html = '';
    let idx = 0;
    ['page', 'action', 'team'].forEach(k => {
      if (!byKind[k]) return;
      html += '<div class="cmdk-group"><div class="cmdk-group-label">' + groups[k] + '</div>';
      byKind[k].forEach(it => {
        const iconHtml = it.icon_img
          ? '<img src="' + it.icon_img + '" alt="">'
          : (it.icon || '');
        html += '<div class="cmdk-item' + (idx === 0 ? ' focused' : '') + '" data-idx="' + idx + '">' +
                  '<div class="icon">' + iconHtml + '</div>' +
                  '<div>' + it.label + '</div>' +
                  '<span class="meta">' + (it.meta || '') + '</span>' +
                '</div>';
        idx++;
      });
      html += '</div>';
    });
    list.innerHTML = html;

    list.querySelectorAll('.cmdk-item').forEach(el => {
      el.addEventListener('click', () => runCmdItem(parseInt(el.dataset.idx, 10)));
    });
  }

  function runCmdItem(i) {
    const it = cmdkState.items[i];
    if (!it) return;
    closeCmdK();
    if (it.run) it.run();
    else if (it.href) window.location = it.href;
  }

  function openCmdK() {
    document.getElementById('cmdk-overlay').classList.add('open');
    const input = document.getElementById('cmdk-input');
    input.value = '';
    renderCmdK('');
    setTimeout(() => input.focus(), 30);
  }

  function closeCmdK() {
    document.getElementById('cmdk-overlay').classList.remove('open');
  }

  // ── Keyboard handling ─────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    const isMod = e.metaKey || e.ctrlKey;
    const isCmdK = isMod && e.key.toLowerCase() === 'k';
    if (isCmdK) { e.preventDefault(); openCmdK(); return; }

    if (e.key === 'Escape') closeCmdK();

    const overlay = document.getElementById('cmdk-overlay');
    if (overlay && overlay.classList.contains('open')) {
      const items = document.querySelectorAll('.cmdk-item');
      if (!items.length) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        cmdkState.focused = (cmdkState.focused + 1) % items.length;
        updateFocus();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        cmdkState.focused = (cmdkState.focused - 1 + items.length) % items.length;
        updateFocus();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        runCmdItem(cmdkState.focused);
      }
    }
  });

  function updateFocus() {
    const items = document.querySelectorAll('.cmdk-item');
    items.forEach((el, i) => el.classList.toggle('focused', i === cmdkState.focused));
    const target = items[cmdkState.focused];
    if (target) target.scrollIntoView({ block: 'nearest' });
  }

  // ── Cmd palette input event ───────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('cmdk-input');
    if (input) {
      input.addEventListener('input', () => renderCmdK(input.value));
    }
    animateProbBars();
    drawSparklines();
  });

  // ── Animate prob bars on load ────────────────────────────
  function animateProbBars() {
    document.querySelectorAll('.prob-bar .seg-l').forEach((seg) => {
      const target = seg.dataset.width;
      if (target) {
        seg.style.width = '0%';
        // Force layout, then transition
        requestAnimationFrame(() => {
          setTimeout(() => { seg.style.width = target + '%'; }, 60);
        });
      }
    });
  }

  // ── Sparklines ────────────────────────────────────────────
  function drawSparklines() {
    document.querySelectorAll('[data-sparkline]').forEach((el) => {
      const raw = el.dataset.sparkline;
      if (!raw) return;
      const pts = raw.split(',').map(Number).filter(n => !isNaN(n));
      if (pts.length < 2) return;
      renderSparkline(el, pts);
    });
  }

  function renderSparkline(el, pts) {
    const w = el.clientWidth || 84, h = el.clientHeight || 22;
    const min = Math.min(...pts), max = Math.max(...pts);
    const range = (max - min) || 1;
    const xs = pts.map((_, i) => (i / (pts.length - 1)) * (w - 2) + 1);
    const ys = pts.map(p => h - 2 - ((p - min) / range) * (h - 4));
    const path = xs.map((x, i) => (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + ys[i].toFixed(1)).join(' ');
    const area = path + ' L' + xs[xs.length-1].toFixed(1) + ',' + h + ' L' + xs[0].toFixed(1) + ',' + h + ' Z';
    const lastTrend = pts[pts.length-1] >= pts[0] ? 'lime' : 'red';
    const cls = el.dataset.trend === 'auto' ? lastTrend : (el.dataset.trend || '');
    const strokeColor = cls === 'lime' ? 'var(--lime)' : cls === 'red' ? 'var(--red)' : 'currentColor';
    el.innerHTML = '' +
      '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" ' +
        'style="width:100%; height:100%; display:block; overflow:visible;">' +
        '<path d="' + area + '" style="fill: currentColor; opacity: 0.18;"/>' +
        '<path d="' + path + '" style="fill:none; stroke:' + strokeColor + '; stroke-width:1.8; stroke-linejoin:round;"/>' +
        '<circle cx="' + xs[xs.length-1] + '" cy="' + ys[ys.length-1] + '" r="2.4" ' +
          'style="fill:' + strokeColor + ';"/>' +
      '</svg>';
  }

  // ── Public ────────────────────────────────────────────────
  window.app = {
    toast,
    toggleBackend,
    openCmdK,
    closeCmdK,
    renderSparkline,
  };
})();
