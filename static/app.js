(() => {
  const $ = (id) => document.getElementById(id);

  let latestState = null;
  let ws = null;
  let wsConnected = false;
  let sweepAngle = 0;

  let localMode = location.protocol === "file:" ? "nominal" : "nominal";
  let localStep = 0;
  const localHistory = [];
  const localEvents = [
    {
      time: new Date().toTimeString().slice(0, 8),
      severity: "INFO",
      message: "360° ultrasonic scan locked: A=15.12 m², H=2.75 m, V=41.58 m³ (100% valid)",
    },
  ];

  const defaultRadii = Array.from({ length: 36 }, (_, i) => {
    const a = (i * 10 * Math.PI) / 180;
    const ca = Math.max(1e-4, Math.abs(Math.cos(a)));
    const sa = Math.max(1e-4, Math.abs(Math.sin(a)));
    return Math.round(Math.min(210 / ca, 180 / sa));
  });

  function setArc(arcId, value, minVal, maxVal) {
    const el = $(arcId);
    if (!el) return;
    const totalLen = 235;
    const pct = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal)));
    el.style.strokeDashoffset = String(totalLen * (1 - pct));
  }

  function setLedAndCard(ledId, textId, cardId, pinLabel, level) {
    const led = $(ledId);
    const txt = $(textId);
    const card = $(cardId);
    led.classList.remove("led-warn", "led-crit");
    card.classList.remove("panel-warn", "panel-crit");
    if (level === 2) {
      led.classList.add("led-crit");
      card.classList.add("panel-crit");
      txt.textContent = `${pinLabel} · CRIT (5Hz)`;
    } else if (level === 1) {
      led.classList.add("led-warn");
      card.classList.add("panel-warn");
      txt.textContent = `${pinLabel} · WARN (ON)`;
    } else {
      txt.textContent = `${pinLabel} · OK`;
    }
  }

  function formatSigned(val, decimals = 1) {
    const n = Number(val || 0);
    return `${n >= 0 ? "+" : ""}${n.toFixed(decimals)}`;
  }

  function syncModeButtons(mode) {
    document.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    if ($("sim-select")) $("sim-select").value = mode;
  }

  function renderRadar(geom) {
    const canvas = $("radar-canvas");
    if (!canvas || !geom || !Array.isArray(geom.radii_cm)) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.max(250, ...geom.radii_cm);
    const radiusPx = Math.min(w, h) * 0.42;
    const scale = radiusPx / maxR;

    ctx.clearRect(0, 0, w, h);

    // Calibrated range rings
    ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
    ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
    ctx.font = "10px monospace";
    for (let ring = 1; ring <= 3; ring++) {
      const r = (radiusPx * ring) / 3;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Cardinal axes
    ctx.beginPath();
    ctx.moveTo(cx - radiusPx * 1.05, cy);
    ctx.lineTo(cx + radiusPx * 1.05, cy);
    ctx.moveTo(cx, cy - radiusPx * 1.05);
    ctx.lineTo(cx, cy + radiusPx * 1.05);
    ctx.stroke();
    ctx.fillText("0°N", cx - 9, cy - radiusPx - 5);
    ctx.fillText("90°E", cx + radiusPx + 4, cy + 3);
    ctx.fillText("180°S", cx - 14, cy + radiusPx + 12);

    // Rotating sonar sweep cone
    sweepAngle = (sweepAngle + 0.06) % (Math.PI * 2);
    const grad = ctx.createRadialGradient(cx, cy, 4, cx, cy, radiusPx);
    grad.addColorStop(0, "rgba(34, 211, 238, 0.28)");
    grad.addColorStop(1, "rgba(34, 211, 238, 0.0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radiusPx, sweepAngle - 0.45, sweepAngle);
    ctx.closePath();
    ctx.fill();

    // Room cross-section polygon
    const radii = geom.radii_cm;
    const n = radii.length;
    if (n > 2) {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const ang = (i * 2 * Math.PI) / n - Math.PI / 2;
        const x = cx + radii[i] * scale * Math.cos(ang);
        const y = cy + radii[i] * scale * Math.sin(ang);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(34, 211, 238, 0.14)";
      ctx.fill();
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#22d3ee";
      for (let i = 0; i < n; i++) {
        const ang = (i * 2 * Math.PI) / n - Math.PI / 2;
        const x = cx + radii[i] * scale * Math.cos(ang);
        const y = cy + radii[i] * scale * Math.sin(ang);
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Center sensor hub
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
    ctx.fill();
  }

  function renderChart(history) {
    const canvas = $("trend-canvas");
    if (!canvas || !Array.isArray(history) || history.length < 2) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const padL = 48;
    const padR = 42;
    const padT = 16;
    const padB = 24;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    ctx.clearRect(0, 0, w, h);

    const co2Vals = history.map((p) => p.co2_ppm);
    const rawVals = history.map((p) => p.co2_raw_ppm);
    const maxCo2 = Math.max(1500, ...co2Vals, ...rawVals) * 1.08;
    const minCo2 = 300;

    // Grid & axis labels
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px monospace";
    for (let i = 0; i <= 4; i++) {
      const y = padT + (ph * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      const label = Math.round(maxCo2 - ((maxCo2 - minCo2) * i) / 4);
      ctx.fillText(`${label}`, 6, y + 4);
      const rhsLabel = Math.round(100 - 25 * i);
      ctx.fillText(`${rhsLabel}`, w - padR + 6, y + 4);
    }

    const xAt = (i) => padL + (pw * i) / (history.length - 1);
    const yCo2 = (v) => padT + ph * (1 - (v - minCo2) / (maxCo2 - minCo2));
    const yPct = (v) => padT + ph * (1 - v / 100.0);

    // Warning threshold line (1200 ppm)
    if (maxCo2 >= 1200) {
      const yWarn = yCo2(1200);
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
      ctx.beginPath();
      ctx.moveTo(padL, yWarn);
      ctx.lineTo(w - padR, yWarn);
      ctx.stroke();
      ctx.restore();
    }

    // Gradient area fill under AI CO2 curve
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = xAt(i);
      const y = yCo2(history[i].co2_ppm);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(xAt(history.length - 1), padT + ph);
    ctx.lineTo(xAt(0), padT + ph);
    ctx.closePath();
    const areaGrad = ctx.createLinearGradient(0, padT, 0, padT + ph);
    areaGrad.addColorStop(0, "rgba(34, 211, 238, 0.25)");
    areaGrad.addColorStop(1, "rgba(34, 211, 238, 0.0)");
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // Spike vertical markers
    for (let i = 0; i < history.length; i++) {
      if (history[i].spike) {
        const x = xAt(i);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.42)";
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + ph);
        ctx.stroke();
      }
    }

    function drawSeries(getter, yMap, color, width = 2, dashed = false) {
      ctx.save();
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const x = xAt(i);
        const y = yMap(getter(history[i]));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    drawSeries((p) => p.co2_raw_ppm, yCo2, "#64748b", 1.5, true);
    drawSeries((p) => p.co2_ppm, yCo2, "#22d3ee", 2.5, false);
    drawSeries((p) => p.temp_c, yPct, "#f59e0b", 1.8, false);
    drawSeries((p) => p.hum_pct, yPct, "#10b981", 1.8, false);
  }

  function renderState(s) {
    latestState = s;
    if (!s || !s.telemetry) return;

    const tel = s.telemetry;
    const cal = s.calibration;
    const tr = s.trends;
    const sp = s.spikes;
    const dec = s.decision;
    const geom = s.geometry;

    $("ai-latency").textContent = `${s.ai_latency_ms} ms`;
    if (s.model_metrics && s.model_metrics.error_reduction_pct !== undefined) {
      $("ml-gain").textContent = `${s.model_metrics.error_reduction_pct}%`;
    }

    const badge = $("link-badge");
    badge.className = "status-pill";
    if (s.sim_mode && s.sim_mode !== "off") {
      badge.classList.add("pill-cyan");
      badge.textContent = `SIMULATOR // ${s.sim_mode.toUpperCase()}`;
    } else if (s.device_online) {
      badge.classList.add("pill-ok");
      badge.textContent = "STM32 + ESP-01 ONLINE";
    } else {
      badge.classList.add("pill-warn");
      badge.textContent = "AWAITING ESP-01 LINK";
    }

    syncModeButtons(s.sim_mode || "off");

    // Channel 1: Gas
    $("val-co2").textContent = Math.round(cal.co2_ppm);
    $("val-co2-raw").textContent = Math.round(cal.co2_raw_ppm);
    $("val-th-factor").textContent = `×${cal.th_correction_factor.toFixed(3)}`;
    $("val-ch4").textContent = Math.round(cal.ch4_ppm);
    $("val-r135").textContent = tel.ratio_135.toFixed(2);
    $("rate-co2").textContent = `${formatSigned(tr.co2.rate_per_min, 1)} ppm/m`;
    $("r2-co2").textContent = tr.co2.r_squared.toFixed(2);
    $("proj-co2").textContent = `${Math.round(tr.co2.projected_5min)} ppm`;
    $("z-co2").textContent = sp.channels.co2.z_score.toFixed(2);
    setArc("arc-co2", cal.co2_ppm, 350, 3500);

    // Channel 2: Temp
    $("val-temp").textContent = tel.temp_c.toFixed(1);
    $("val-sound").textContent = `${(331.3 + 0.606 * tel.temp_c).toFixed(1)} m/s`;
    $("rate-temp").textContent = `${formatSigned(tr.temp.rate_per_min, 2)} °C/m`;
    $("r2-temp").textContent = tr.temp.r_squared.toFixed(2);
    $("proj-temp").textContent = `${tr.temp.projected_5min.toFixed(1)} °C`;
    $("z-temp").textContent = sp.channels.temp.z_score.toFixed(2);
    setArc("arc-temp", tel.temp_c, 0, 50);

    // Channel 3: Humidity
    $("val-hum").textContent = tel.hum_pct.toFixed(1);
    const dew = tel.temp_c - (100 - tel.hum_pct) / 5;
    $("val-dew").textContent = `${dew.toFixed(1)} °C`;
    $("rate-hum").textContent = `${formatSigned(tr.hum.rate_per_min, 2)} %/m`;
    $("r2-hum").textContent = tr.hum.r_squared.toFixed(2);
    $("proj-hum").textContent = `${tr.hum.projected_5min.toFixed(1)} %`;
    $("z-hum").textContent = sp.channels.hum.z_score.toFixed(2);
    setArc("arc-hum", tel.hum_pct, 0, 100);

    // Hardware LEDs & card glow
    setLedAndCard("led-gas", "led-gas-text", "card-gas", "PB10", dec.lvl_gas);
    setLedAndCard("led-temp", "led-temp-text", "card-temp", "PB0", dec.lvl_temp);
    setLedAndCard("led-hum", "led-hum-text", "card-hum", "PB1", dec.lvl_hum);

    const deck = $("command-deck");
    const overallBadge = $("overall-badge");
    const dot = $("overall-dot");
    deck.classList.remove("deck-warn", "deck-crit");
    overallBadge.className = "status-pill";

    if (dec.buzzer_mode === 2) {
      deck.classList.add("deck-crit");
      overallBadge.classList.add("pill-crit");
      overallBadge.textContent = "STATE: CRITICAL ALARM";
      dot.style.background = "#ef4444";
      dot.style.boxShadow = "0 0 18px #ef4444";
      $("buzzer-pill").textContent = tel.muted
        ? "BUZZER (PA6): MUTED BY OPERATOR"
        : "BUZZER (PA6): RAPID CRITICAL ALARM";
    } else if (dec.buzzer_mode === 1) {
      deck.classList.add("deck-warn");
      overallBadge.classList.add("pill-warn");
      overallBadge.textContent = "STATE: WARNING";
      dot.style.background = "#f59e0b";
      dot.style.boxShadow = "0 0 16px #f59e0b";
      $("buzzer-pill").textContent = tel.muted
        ? "BUZZER (PA6): MUTED BY OPERATOR"
        : "BUZZER (PA6): INTERMITTENT CHIRP";
    } else {
      overallBadge.classList.add("pill-nominal");
      overallBadge.textContent = "STATE: NOMINAL";
      dot.style.background = "#10b981";
      dot.style.boxShadow = "0 0 16px #10b981";
      $("buzzer-pill").textContent = "BUZZER (PA6): SILENT";
    }

    if (geom) {
      $("scan-badge").textContent = `36 ANGLES · ${geom.valid_pct}% VALID`;
      $("geom-area").textContent = `${geom.area_m2.toFixed(2)} m²`;
      $("geom-height").textContent = `${geom.height_m.toFixed(2)} m`;
      $("geom-volume").textContent = `${geom.volume_m3.toFixed(2)} m³`;
      $("val-vol-factor").textContent = cal.volume_mixing_factor.toFixed(3);
      $("mass-co2").textContent = `${cal.co2_excess_mass_g.toFixed(1)} g (+${cal.co2_leak_rate_g_min.toFixed(2)} g/min)`;
      $("vent-req").textContent = `${cal.required_ventilation_m3_h.toFixed(1)} m³/h (${cal.recommended_ach.toFixed(1)} ACH)`;

      if (cal.time_to_crit_min !== null) {
        const ttw = cal.time_to_warn_min !== null ? `${cal.time_to_warn_min}m warn / ` : "";
        $("time-to-crit").textContent = `${ttw}${cal.time_to_crit_min}m to CRIT`;
      } else {
        $("time-to-crit").textContent = "∞ (Stable)";
      }
      renderRadar(geom);
    }

    renderChart(s.history || []);

    $("decision-headline").textContent = dec.headline;
    const advWrap = $("advisory-list");
    advWrap.innerHTML = "";
    for (const adv of dec.advisories || []) {
      const div = document.createElement("div");
      div.className = `advisory-item ${adv.priority}`;
      const stepsHtml = (adv.containment_steps || [])
        .map((st) => `<li>${st}</li>`)
        .join("");
      div.innerHTML = `
        <div class="advisory-title">[${adv.priority} · ${adv.channel}] ${adv.title}</div>
        <ol class="advisory-steps">${stepsHtml}</ol>
      `;
      advWrap.appendChild(div);
    }

    const logEl = $("event-log");
    logEl.innerHTML = "";
    for (const ev of s.events || []) {
      const li = document.createElement("li");
      li.textContent = `[${ev.time}] (${ev.severity}) ${ev.message}`;
      logEl.appendChild(li);
    }
  }

  function stepLocalPreview() {
    if (wsConnected) return;
    const mode = localMode === "off" ? "nominal" : localMode;
    localStep++;
    const vol = 41.58;
    let co2 = 432 + Math.sin(localStep * 0.25) * 4;
    let temp = 24.2 + Math.cos(localStep * 0.15) * 0.08;
    let hum = 48.0 + Math.sin(localStep * 0.1) * 0.2;
    let rateCo2 = 0.0;
    let rateTemp = 0.0;
    let r2Co2 = 0.08;
    let zCo2 = 0.35;
    let spike = false;

    if (mode === "gradual_co2") {
      co2 = Math.min(3100, 440 + localStep * 14);
      rateCo2 = 1680.0;
      r2Co2 = 0.99;
      zCo2 = 1.4;
    } else if (mode === "spike_gas") {
      if (localStep > 4) {
        co2 = Math.min(3600, 435 + 1950 * (1 - Math.exp(-(localStep - 4) * 0.45)));
        spike = localStep <= 9;
        zCo2 = spike ? 8.4 : 2.1;
        rateCo2 = 940.0;
        r2Co2 = 0.84;
      }
    } else if (mode === "thermal_fire") {
      if (localStep > 4) {
        co2 = Math.min(3800, 440 + (localStep - 4) * 165);
        temp = Math.min(45.0, 24.4 + (localStep - 4) * 1.15);
        hum = Math.max(19.0, 48.0 - (localStep - 4) * 1.3);
        spike = true;
        zCo2 = 6.8;
        rateCo2 = 1980.0;
        rateTemp = 13.8;
        r2Co2 = 0.96;
      }
    }

    const rawCo2 = Math.round(co2 * 1.12);
    const ratio135 = Math.pow(co2 / 110.47, 1 / -2.862);
    const ch4 = ratio135 < 0.52 ? Math.round(12 * Math.pow(ratio135, -3.25)) : 2;
    const excessG = Math.max(0, ((co2 - 420) * 1e-6 * 101325 * vol * 44.01) / (8.314 * (temp + 273.15)));
    const leakGMin = Math.max(0, (rateCo2 * 1e-6 * 101325 * vol * 44.01) / (8.314 * (temp + 273.15)));
    const ach = co2 > 1200 ? 10.5 : 2.0;

    localHistory.push({
      ts: Date.now() / 1000,
      co2_ppm: Math.round(co2 * 10) / 10,
      co2_raw_ppm: rawCo2,
      temp_c: Math.round(temp * 10) / 10,
      hum_pct: Math.round(hum * 10) / 10,
      spike,
    });
    if (localHistory.length > 80) localHistory.shift();

    const lvlGas = co2 >= 2500 || spike ? 2 : co2 >= 1200 || mode === "gradual_co2" ? 1 : 0;
    const lvlTemp = temp >= 40 ? 2 : temp >= 32 ? 1 : 0;
    const lvlHum = hum <= 25 ? 1 : 0;
    const buz = Math.max(lvlGas, lvlTemp, lvlHum);

    const advisories =
      buz === 0
        ? [
            {
              priority: "INFO",
              channel: "SYSTEM",
              title: "All Environmental & Gas Channels Nominal",
              containment_steps: [
                `Room volume (${vol} m³) is within safe air-quality limits (CO₂ ${Math.round(co2)} ppm, T ${temp.toFixed(1)} °C, RH ${hum.toFixed(1)} %).`,
                `Baseline fresh-air exchange of ${(vol * ach).toFixed(1)} m³/h (${ach.toFixed(1)} ACH) maintains steady-state air quality.`,
              ],
            },
          ]
        : [
            {
              priority: buz === 2 ? "CRITICAL" : "WARNING",
              channel: lvlTemp > 0 ? "TEMP" : "GAS",
              title: spike
                ? `Sudden Gas / Thermal Spike Detected (${Math.round(co2)} ppm CO₂, ${temp.toFixed(1)} °C)`
                : `Gradual CO₂ Accumulation (+${rateCo2.toFixed(0)} ppm/min) in ${vol} m³ Room`,
              containment_steps: [
                `Supply ${(vol * ach).toFixed(1)} m³/h of outside air (${ach.toFixed(1)} ACH) to purge the ${vol} m³ room volume.`,
                "Inspect gas valves, compressed CO₂ lines, combustion heaters, and electrical panels immediately.",
                "Evacuate occupants to fresh air if concentration exceeds 2500 ppm or dizziness occurs.",
              ],
            },
          ];

    renderState({
      ai_latency_ms: 0.42,
      model_metrics: { error_reduction_pct: 92.8 },
      sim_mode: localMode,
      device_online: false,
      telemetry: {
        temp_c: temp,
        hum_pct: hum,
        ratio_135: ratio135,
        muted: false,
      },
      calibration: {
        co2_ppm: co2,
        co2_raw_ppm: rawCo2,
        ch4_ppm: ch4,
        th_correction_factor: 0.964,
        volume_mixing_factor: 1.008,
        co2_excess_mass_g: excessG,
        co2_leak_rate_g_min: leakGMin,
        required_ventilation_m3_h: vol * ach,
        recommended_ach: ach,
        time_to_warn_min: rateCo2 > 0 && co2 < 1200 ? Number(((1200 - co2) / rateCo2).toFixed(1)) : null,
        time_to_crit_min: rateCo2 > 0 && co2 < 2500 ? Number(((2500 - co2) / rateCo2).toFixed(1)) : null,
      },
      trends: {
        co2: { rate_per_min: rateCo2, r_squared: r2Co2, projected_5min: co2 + rateCo2 * 5 },
        temp: { rate_per_min: rateTemp, r_squared: rateTemp > 0 ? 0.94 : 0.05, projected_5min: temp + rateTemp * 5 },
        hum: { rate_per_min: 0.0, r_squared: 0.04, projected_5min: hum },
      },
      spikes: {
        channels: {
          co2: { z_score: zCo2 },
          temp: { z_score: rateTemp > 0 ? 5.2 : 0.2 },
          hum: { z_score: 0.25 },
        },
      },
      decision: {
        lvl_gas: lvlGas,
        lvl_temp: lvlTemp,
        lvl_hum: lvlHum,
        buzzer_mode: buz,
        headline: advisories[0].title,
        advisories,
      },
      geometry: {
        area_m2: 15.12,
        height_m: 2.75,
        volume_m3: vol,
        valid_pct: 100,
        radii_cm: defaultRadii,
      },
      history: localHistory,
      events: localEvents,
    });
  }

  function exportCsv() {
    const hist = (latestState && latestState.history) || localHistory;
    if (!hist.length) return;
    const rows = ["timestamp,co2_compensated_ppm,co2_raw_ppm,temp_c,humidity_pct,spike"];
    hist.forEach((p) => {
      rows.push(`${p.ts},${p.co2_ppm},${p.co2_raw_ppm},${p.temp_c},${p.hum_pct},${p.spike ? 1 : 0}`);
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gassentinel_telemetry_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function connectWebSocket() {
    if (location.protocol === "file:") return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/dashboard`);
    ws.onopen = () => {
      wsConnected = true;
    };
    ws.onmessage = (ev) => {
      try {
        renderState(JSON.parse(ev.data));
      } catch (_) {}
    };
    ws.onclose = () => {
      wsConnected = false;
      setTimeout(connectWebSocket, 2000);
    };
  }

  function sendAction(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    } else {
      localEvents.unshift({
        time: new Date().toTimeString().slice(0, 8),
        severity: "INFO",
        message: `Operator executed '${payload.action.toUpperCase()}' command.`,
      });
    }
  }

  function switchMode(mode) {
    localMode = mode;
    localStep = 0;
    syncModeButtons(mode);
    if (location.protocol !== "file:") {
      fetch(`/api/simulate/${mode}`, { method: "POST" }).catch(() => {});
      sendAction({ action: "simulate", mode });
    }
  }

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  $("sim-select").addEventListener("change", (e) => switchMode(e.target.value));
  $("btn-scan").addEventListener("click", () => sendAction({ action: "scan" }));
  $("btn-cal").addEventListener("click", () => sendAction({ action: "calibrate" }));
  $("btn-mute").addEventListener("click", () => sendAction({ action: "mute" }));
  $("btn-export").addEventListener("click", exportCsv);

  if (location.protocol !== "file:") {
    fetch("/api/state")
      .then((r) => r.json())
      .then((s) => renderState(s))
      .catch(() => {});
    connectWebSocket();
  }

  for (let i = 0; i < 14; i++) stepLocalPreview();
  setInterval(stepLocalPreview, 500);
})();
