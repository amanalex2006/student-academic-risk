"use strict";

const $ = (selector) => document.querySelector(selector);
const page = document.body.dataset.page;
const form = $("#student-form");
const numericFields = new Set(["studytime", "failures", "traveltime", "absences", "G1", "G2"]);
const baseFields = ["studytime", "absences", "failures", "traveltime", "schoolsup", "famsup", "activities", "higher", "internet"];
const subjectLabels = {math: "Mathematics", portuguese: "Portuguese"};
const modeLabels = {without_grades: "Without previous grades", with_grades: "With G1 and G2"};
const algorithmLabels = {logistic_regression: "Logistic regression", calibrated_random_forest: "Random forest · sigmoid calibrated", calibrated_decision_tree: "Decision tree · sigmoid calibrated", prior_baseline: "Prior probability baseline"};
const metricLabels = {accuracy: "Accuracy", failure_precision: "Failure precision", failure_recall: "Failure recall", failure_f1: "Failure F1", roc_auc: "ROC-AUC", pr_average_precision: "PR average precision", log_loss: "Log loss ↓", brier_score: "Brier score ↓"};
const percentage = (value) => `${(value * 100).toFixed(1)}%`;
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[char]);
let catalog = [];
let controller = null;
let requestVersion = 0;

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The request could not be completed.");
  return body;
}

function clearResult() {
  requestVersion += 1;
  controller?.abort();
  controller = null;
  if (!form) return;
  $("#empty-result").hidden = false;
  $("#loading-result").hidden = true;
  $("#prediction-result").hidden = true;
  $("#form-error").hidden = true;
  $("#predict-button").disabled = catalog.length !== 4;
  $("#predict-button").innerHTML = 'Estimate performance <span aria-hidden="true">↗</span>';
  $("#result-tag").textContent = "Awaiting profile";
  $("#result-tag").className = "result-tag";
  $("#prediction-notes").replaceChildren();
}

function syncGradeFields() {
  const enabled = form.elements.mode.value === "with_grades";
  document.querySelectorAll(".grade-field").forEach((label) => {
    label.hidden = !enabled;
    const input = label.querySelector("input");
    input.disabled = !enabled;
    input.required = enabled;
  });
}

function updateModelContext() {
  if (!form) return;
  const current = catalog.find((model) => model.subject === form.elements.subject.value && model.mode === form.elements.mode.value);
  if (!current) {
    $("#selected-model-name").textContent = "No saved model available for this selection.";
    for (const field of ["recall", "auc", "records"]) $(`#context-${field}`).textContent = "—";
    return;
  }
  $("#selected-model-name").textContent = algorithmLabels[current.model_name] || current.model_name;
  $("#context-recall").textContent = percentage(current.metrics.failure_recall);
  $("#context-auc").textContent = current.metrics.roc_auc.toFixed(3);
  $("#context-records").textContent = current.test_records;
  $("#context-limit").textContent = current.mode === "without_grades"
    ? "Without prior grades, this model misses many failing students at the 50% threshold. These are retrospective estimates."
    : "Measured on held-out records. Both earlier-period grades must be available before the final grade.";
}

function renderPrediction(result) {
  const pass = result.most_likely_outcome === "Pass";
  $("#empty-result").hidden = true;
  $("#loading-result").hidden = true;
  $("#prediction-result").hidden = false;
  $("#result-tag").textContent = "Estimate ready";
  $("#result-tag").className = `result-tag ${pass ? "pass-tag" : "fail-tag"}`;
  $("#outcome").textContent = pass ? "Likely to pass" : "At risk of failing";
  $("#outcome").className = pass ? "" : "fail-outcome";
  $("#outcome-context").textContent = `${subjectLabels[result.subject]} · ${modeLabels[result.mode]}`;
  for (const outcome of ["pass", "fail"]) {
    $(`#${outcome}-probability`).textContent = percentage(result.probabilities[outcome]);
    $(`#${outcome}-bar`).style.width = percentage(result.probabilities[outcome]);
  }
  const factor = result.prominent_factor;
  $("#factor-name").textContent = factor ? factor.label : "No measurable input contribution";
  $("#factor-description").textContent = factor
    ? `${factor.direction[0].toUpperCase() + factor.direction.slice(1)} by ${Math.abs(factor.percentage_points).toFixed(2)} percentage points relative to the explanation background.`
    : "This model estimates the outcome prior rather than using individual inputs.";
  const maximum = Math.max(.01, ...result.feature_contributions.map((f) => Math.abs(f.percentage_points)));
  $("#factor-bars").innerHTML = result.feature_contributions.map((item, index) => `
    <div class="factor-row ${index === 0 && factor ? "prominent" : ""}">
      <span>${escapeHtml(item.label)}</span>
      <div class="factor-track" aria-hidden="true"><span class="${item.contribution > 0 ? "positive" : ""}" style="width:${Math.abs(item.percentage_points) / maximum * 100}%"></span></div>
      <strong aria-label="${escapeHtml(item.direction)} by ${Math.abs(item.percentage_points).toFixed(2)} percentage points">${item.percentage_points > 0 ? "+" : ""}${item.percentage_points.toFixed(2)} pp</strong>
    </div>`).join("");
  const notes = [...result.notes];
  if (result.is_probability_tie) notes.push("Equal likelihoods: the 50% failure threshold flags this outcome as Fail.");
  $("#prediction-notes").hidden = notes.length === 0;
  $("#prediction-notes").textContent = notes.join(" ");
  const change = result.feature_contributions.reduce((total, item) => total + item.percentage_points, 0);
  $("#explanation-note").textContent = `Background risk ${percentage(result.base_failure_probability)} + contributions ${change >= 0 ? "+" : ""}${change.toFixed(2)} pp = failure likelihood ${percentage(result.probabilities.fail)}. Estimates describe model behavior, not certainty.`;
}

function collectInputs() {
  const mode = form.elements.mode.value;
  const fields = mode === "with_grades" ? [...baseFields, "G1", "G2"] : baseFields;
  const inputs = {};
  for (const name of fields) {
    const value = form.elements[name].value;
    if (!value.trim()) throw new Error("Complete every active field before estimating performance.");
    inputs[name] = numericFields.has(name) ? Number(value) : value;
  }
  return {subject: form.elements.subject.value, mode, inputs};
}

async function predictProfile(payload) {
  clearResult();
  const version = requestVersion;
  controller = new AbortController();
  $("#empty-result").hidden = true;
  $("#loading-result").hidden = false;
  $("#result-tag").textContent = "Calculating";
  $("#predict-button").disabled = true;
  $("#predict-button").textContent = "Calculating…";
  try {
    const result = await getJson("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload), signal: controller.signal});
    if (version !== requestVersion) throw new Error("The profile changed before the estimate completed.");
    renderPrediction(result);
    return result;
  } catch (error) {
    if (error.name !== "AbortError" && version === requestVersion) {
      $("#form-error").textContent = error.message;
      $("#form-error").hidden = false;
      $("#empty-result").hidden = false;
      $("#loading-result").hidden = true;
      $("#result-tag").textContent = "Check profile";
    }
    throw error;
  } finally {
    if (version === requestVersion) {
      $("#predict-button").disabled = catalog.length !== 4;
      $("#predict-button").innerHTML = 'Estimate performance <span aria-hidden="true">↗</span>';
    }
  }
}

function showModels() {
  const grid = $("#model-grid");
  if (!grid) return;
  if (!catalog.length) {
    grid.innerHTML = '<div class="panel library-note"><h2>No models saved yet</h2><p>Run the training notebook through “Save models for the website”, then restart the website.</p></div>';
    return;
  }
  grid.innerHTML = catalog.map((model) => {
    const date = new Date(model.trained_at).toLocaleString(undefined, {year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
    const rows = Object.entries(metricLabels).map(([metric, label]) => `<tr><td>${label}</td><td>${model.metrics[metric].toFixed(3)}</td><td>${model.baseline_metrics[metric].toFixed(3)}</td></tr>`).join("");
    return `<article class="model-card" data-model-id="${escapeHtml(model.id)}">
      <div class="model-card-top"><span class="subject-icon" aria-hidden="true">${model.subject === "math" ? "∑" : "Aa"}</span><span class="ready-badge">Saved &amp; ready</span></div>
      <h2>${escapeHtml(subjectLabels[model.subject])}</h2><p class="mode-description">${escapeHtml(modeLabels[model.mode])}</p>
      <span class="algorithm">${escapeHtml(algorithmLabels[model.model_name] || model.model_name)}</span>
      <div class="model-scores"><div><strong>${percentage(model.metrics.accuracy)}</strong><span>Accuracy</span></div><div><strong>${percentage(model.metrics.failure_recall)}</strong><span>Failure recall</span></div><div><strong>${model.metrics.roc_auc.toFixed(3)}</strong><span>ROC-AUC</span></div></div>
      <div class="model-file"><code>${escapeHtml(model.filename)}</code><span>${(model.size_bytes / 1024).toFixed(1)} KB</span></div>
      <p class="model-provenance">Saved ${escapeHtml(date)} · ${model.features.length} inputs · ${model.train_records} train / ${model.test_records} test records</p>
      <details><summary>Evaluation results &amp; model details</summary><table class="metrics-table"><thead><tr><th>Metric</th><th>Selected model</th><th>Prior baseline</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="model-checksum">Training CV log loss: ${model.cv_log_loss.toFixed(3)} · Failure threshold: 50%<br>scikit-learn ${escapeHtml(model.packages["scikit-learn"])}<br>SHA-256: ${escapeHtml(model.sha256)}</p></details>
    </article>`;
  }).join("");
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    try { await predictProfile(collectInputs()); } catch (_) { /* The visible error has already been set. */ }
  });
  form.addEventListener("input", clearResult);
  form.addEventListener("change", () => { clearResult(); syncGradeFields(); updateModelContext(); });
  form.addEventListener("reset", () => { clearResult(); setTimeout(() => { syncGradeFields(); updateModelContext(); }, 0); });
  $("#sample-button").addEventListener("click", () => {
    clearResult();
    const sample = {studytime: 2, absences: 6, failures: 0, traveltime: 2, schoolsup: "no", famsup: "yes", activities: "yes", higher: "yes", internet: "yes", G1: 8, G2: 9};
    for (const [name, value] of Object.entries(sample)) form.elements[name].value = value;
    syncGradeFields();
  });
  $("#predict-button").disabled = true;
}

async function initialize() {
  try {
    const result = await getJson("/api/models");
    catalog = result.models;
    $("#nav-count").textContent = String(catalog.length).padStart(2, "0");
    $("#model-status").innerHTML = `<span></span>${catalog.length} models ${catalog.length === 4 ? "ready" : "saved"}`;
    if (!catalog.length) {
      $("#model-status").classList.add("unavailable");
      $("#global-error").textContent = "No trained models are saved yet. Run the training notebook through its model export section, then restart the website.";
      $("#global-error").hidden = false;
    }
    if (form) { syncGradeFields(); updateModelContext(); clearResult(); }
    showModels();
  } catch (error) {
    $("#model-status").textContent = "Models unavailable";
    $("#model-status").classList.add("unavailable");
    $("#global-error").textContent = error.message;
    $("#global-error").hidden = false;
    if ($("#model-grid")) $("#model-grid").textContent = "The saved model registry could not be loaded.";
  }
  registerBrowserTools();
}

function registerBrowserTools() {
  if (!document.modelContext?.registerTool) return;
  const tools = [{
    name: "read_saved_risk_models", title: "Read saved academic risk models",
    description: "Read the saved model catalog and measured course evaluation results displayed in the model library.",
    inputSchema: {type: "object", properties: {}, additionalProperties: false},
    annotations: {readOnlyHint: true},
    execute: async () => catalog.map(({id, model_name, metrics}) => ({id, model_name, metrics})),
  }];
  if (form) tools.push({
    name: "predict_course_outcome", title: "Estimate a student's course outcome",
    description: "Enter a student profile in the assessment form, compute fail/pass probabilities with its saved model, and show individual contributions.",
    inputSchema: {type: "object", properties: {
      subject: {type: "string", enum: ["math", "portuguese"]},
      mode: {type: "string", enum: ["without_grades", "with_grades"]},
      inputs: {type: "object", properties: {
        studytime: {type: "integer", minimum: 1, maximum: 4}, absences: {type: "integer", minimum: 0, maximum: 93},
        failures: {type: "integer", minimum: 0, maximum: 3}, traveltime: {type: "integer", minimum: 1, maximum: 4},
        ...Object.fromEntries(["schoolsup", "famsup", "activities", "higher", "internet"].map((name) => [name, {type: "string", enum: ["yes", "no"]}])),
        G1: {type: "integer", minimum: 0, maximum: 20}, G2: {type: "integer", minimum: 0, maximum: 20},
      }, required: baseFields, additionalProperties: false},
    }, required: ["subject", "mode", "inputs"], additionalProperties: false},
    annotations: {readOnlyHint: false},
    execute: async (payload) => {
      if (!subjectLabels[payload.subject] || !modeLabels[payload.mode]) throw new Error("Invalid subject or mode.");
      clearResult();
      form.reset();
      form.elements.subject.value = payload.subject;
      form.elements.mode.value = payload.mode;
      for (const [field, value] of Object.entries(payload.inputs)) if (form.elements[field]) form.elements[field].value = value;
      syncGradeFields(); updateModelContext();
      return predictProfile(payload);
    },
  });
  for (const tool of tools) {
    try { Promise.resolve(document.modelContext.registerTool(tool)).catch(() => {}); }
    catch (_) { /* Browser tool support is optional; the visible form remains available. */ }
  }
}

initialize();
