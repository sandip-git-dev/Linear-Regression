const sampleCsv = `Hours,Attendance,Assignments,Department,Marks
2,70,4,Mech,48
3,75,5,CSE,55
4,82,6,ECE,61
5,85,7,CSE,68
6,88,8,EEE,72
7,90,8,CSE,78
8,92,9,Mech,83
1,65,3,Civil,42
3,78,5,ECE,57
4,80,6,CSE,62
5,86,7,EEE,70
6,89,8,CSE,75
7,91,9,Civil,81
8,94,10,ECE,87
2,72,4,Mech,50
9,96,10,CSE,91
5,,7,ECE,69
4,84,,Civil,63
6,87,8,EEE,74
7,93,9,CSE,84`;

const state = {
  rawRows: [],
  columns: [],
  columnTypes: {},
  targetColumn: "",
  featureColumns: [],
  preprocessConfig: {},
  processedRows: [],
  processedFeatureNames: [],
  targetValues: [],
  preprocessArtifacts: null,
  trainResult: null
};

const els = {
  csvFile: document.getElementById("csvFile"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  datasetSummary: document.getElementById("datasetSummary"),
  datasetPreview: document.getElementById("datasetPreview"),
  targetColumn: document.getElementById("targetColumn"),
  missingStrategy: document.getElementById("missingStrategy"),
  encodingStrategy: document.getElementById("encodingStrategy"),
  scalingStrategy: document.getElementById("scalingStrategy"),
  featureSelection: document.getElementById("featureSelection"),
  runPreprocessBtn: document.getElementById("runPreprocessBtn"),
  beforeAfterComparison: document.getElementById("beforeAfterComparison"),
  processedPreview: document.getElementById("processedPreview"),
  edaFeature: document.getElementById("edaFeature"),
  histogramChart: document.getElementById("histogramChart"),
  scatterChart: document.getElementById("scatterChart"),
  correlationMatrix: document.getElementById("correlationMatrix"),
  edaNarrative: document.getElementById("edaNarrative"),
  formulaPanel: document.getElementById("formulaPanel"),
  workedSample: document.getElementById("workedSample"),
  regressionModes: document.getElementById("regressionModes"),
  intermediateValues: document.getElementById("intermediateValues"),
  splitRatio: document.getElementById("splitRatio"),
  splitValue: document.getElementById("splitValue"),
  kFolds: document.getElementById("kFolds"),
  learningRate: document.getElementById("learningRate"),
  iterations: document.getElementById("iterations"),
  trainModelBtn: document.getElementById("trainModelBtn"),
  parameterTable: document.getElementById("parameterTable"),
  costChart: document.getElementById("costChart"),
  regressionChart: document.getElementById("regressionChart"),
  predictionComparison: document.getElementById("predictionComparison"),
  predictionForm: document.getElementById("predictionForm"),
  predictionExplanation: document.getElementById("predictionExplanation"),
  mseMetric: document.getElementById("mseMetric"),
  maeMetric: document.getElementById("maeMetric"),
  r2Metric: document.getElementById("r2Metric"),
  cvResults: document.getElementById("cvResults"),
  pipelineNav: document.getElementById("pipelineNav"),
  statusList: document.getElementById("statusList")
};

init();

function init() {
  renderPipelineNav();
  renderStatus();
  renderFormulaPanel();
  renderRegressionModes();
  els.csvFile.addEventListener("change", handleFileUpload);
  els.loadSampleBtn.addEventListener("click", () => loadDatasetFromCsv(sampleCsv, "sample.csv"));
  els.runPreprocessBtn.addEventListener("click", runPreprocessing);
  els.edaFeature.addEventListener("change", renderEda);
  els.splitRatio.addEventListener("input", () => {
    els.splitValue.textContent = `${els.splitRatio.value}%`;
  });
  els.trainModelBtn.addEventListener("click", trainModel);
}

function renderPipelineNav() {
  const panels = [...document.querySelectorAll(".panel")];
  els.pipelineNav.innerHTML = panels.map((panel) => {
    const summary = panel.querySelector("summary span").textContent;
    return `<a href="#" class="nav-link" data-stage-link="${panel.dataset.stage}">${summary}</a>`;
  }).join("");

  document.querySelectorAll("[data-stage-link]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const stage = event.currentTarget.dataset.stageLink;
      const panel = document.querySelector(`.panel[data-stage="${stage}"]`);
      panel.open = true;
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveNav(stage);
    });
  });

  document.querySelectorAll(".panel").forEach((panel) => {
    panel.addEventListener("toggle", () => {
      if (panel.open) setActiveNav(panel.dataset.stage);
    });
  });

  setActiveNav("dataset");
}

function setActiveNav(stage) {
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle("active", link.dataset.stageLink === stage);
  });
}

function renderStatus() {
  const items = [
    { label: "Dataset loaded", done: state.rawRows.length > 0 },
    { label: "Preprocessing complete", done: state.processedRows.length > 0 },
    { label: "Model trained", done: Boolean(state.trainResult) },
    { label: "Prediction ready", done: Boolean(state.trainResult) }
  ];
  els.statusList.innerHTML = items.map((item) => `<li>${item.done ? "Complete" : "Pending"}: ${item.label}</li>`).join("");
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => loadDatasetFromCsv(reader.result, file.name);
  reader.readAsText(file);
}

function loadDatasetFromCsv(csvText, filename) {
  const rows = parseCsv(csvText);
  if (!rows.length) {
    alert("The CSV file could not be parsed.");
    return;
  }

  state.rawRows = rows;
  state.columns = Object.keys(rows[0]);
  state.columnTypes = inferColumnTypes(rows, state.columns);
  state.targetColumn = state.columns[state.columns.length - 1];
  state.featureColumns = state.columns.slice(0, -1);
  state.processedRows = [];
  state.processedFeatureNames = [];
  state.targetValues = [];
  state.preprocessArtifacts = null;
  state.trainResult = null;

  renderDatasetSummary(filename);
  els.datasetPreview.innerHTML = renderTable(state.rawRows.slice(0, 6), state.columns);
  populateTargetSelector();
  populateFeatureSelection();
  resetDownstreamViews();
  renderStatus();
}

function parseCsv(csvText) {
  const rows = [];
  const lines = [];
  let current = "";
  let row = [];
  let inQuotes = false;

  for (let i = 0; i < csvText.length; i += 1) {
    const char = csvText[i];
    const next = csvText[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      row.push(current.trim());
      current = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(current.trim());
      current = "";
      if (row.some((cell) => cell !== "")) lines.push(row);
      row = [];
    } else {
      current += char;
    }
  }

  if (current.length || row.length) {
    row.push(current.trim());
    lines.push(row);
  }

  if (lines.length < 2) return [];
  const headers = lines[0];

  for (let i = 1; i < lines.length; i += 1) {
    const values = lines[i];
    const item = {};
    headers.forEach((header, index) => {
      item[header] = values[index] ?? "";
    });
    rows.push(item);
  }

  return rows;
}

function inferColumnTypes(rows, columns) {
  const types = {};
  columns.forEach((column) => {
    let numericCount = 0;
    let filledCount = 0;

    rows.forEach((row) => {
      const value = row[column];
      if (value !== "") {
        filledCount += 1;
        if (!Number.isNaN(Number(value))) numericCount += 1;
      }
    });

    types[column] = filledCount > 0 && numericCount === filledCount ? "numeric" : "categorical";
  });
  return types;
}

function renderDatasetSummary(filename) {
  const missingCount = state.columns.reduce((acc, column) => {
    return acc + state.rawRows.filter((row) => row[column] === "").length;
  }, 0);

  const typeBadges = state.columns.map((column) => {
    return `<span class="chip">${column}: ${state.columnTypes[column]}</span>`;
  }).join("");

  els.datasetSummary.innerHTML = `
    <p><strong>File:</strong> ${filename}</p>
    <p><strong>Shape:</strong> ${state.rawRows.length} rows x ${state.columns.length} columns</p>
    <p><strong>Missing cells:</strong> ${missingCount}</p>
    <div>${typeBadges}</div>
  `;
}

function populateTargetSelector() {
  els.targetColumn.innerHTML = state.columns.map((column) => {
    const selected = column === state.targetColumn ? "selected" : "";
    return `<option value="${column}" ${selected}>${column}</option>`;
  }).join("");

  els.targetColumn.onchange = () => {
    state.targetColumn = els.targetColumn.value;
    populateFeatureSelection();
  };
}

function populateFeatureSelection() {
  const features = state.columns.filter((column) => column !== state.targetColumn);
  state.featureColumns = [...features];
  els.featureSelection.innerHTML = features.map((feature) => `
    <label class="check-option">
      <input type="checkbox" value="${feature}" checked>
      <span>${feature}</span>
    </label>
  `).join("");
}

function resetDownstreamViews() {
  els.beforeAfterComparison.innerHTML = "Run preprocessing to compare the transformed dataset.";
  els.processedPreview.innerHTML = "Processed data will appear here.";
  els.edaFeature.innerHTML = "";
  resetTrainingOutputs();
}

function runPreprocessing() {
  if (!state.rawRows.length) {
    alert("Upload a dataset first.");
    return;
  }

  const selectedFeatures = [...els.featureSelection.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
  if (!selectedFeatures.length) {
    alert("Select at least one feature.");
    return;
  }

  const config = {
    targetColumn: els.targetColumn.value,
    features: selectedFeatures,
    missingStrategy: els.missingStrategy.value,
    encodingStrategy: els.encodingStrategy.value,
    scalingStrategy: els.scalingStrategy.value
  };

  const result = preprocessData(state.rawRows, state.columnTypes, config);
  if (!result.ok) {
    alert(result.message);
    return;
  }

  state.targetColumn = config.targetColumn;
  state.featureColumns = selectedFeatures;
  state.preprocessConfig = config;
  state.processedRows = result.rows;
  state.processedFeatureNames = result.featureNames;
  state.targetValues = result.targetValues;
  state.preprocessArtifacts = result.artifacts;
  state.trainResult = null;

  renderBeforeAfterComparison(result);
  els.processedPreview.innerHTML = renderTable(result.rows.slice(0, 6), [...result.featureNames, state.targetColumn]);
  populateEdaFeatureSelector();
  renderEda();
  resetModelOutputs();
  renderStatus();
}

function preprocessData(rawRows, columnTypes, config) {
  let rows = rawRows.map((row) => ({ ...row }));
  if (columnTypes[config.targetColumn] !== "numeric") {
    return { ok: false, message: "The target column must be numeric for linear regression." };
  }

  if (config.missingStrategy === "drop") {
    rows = rows.filter((row) => [...config.features, config.targetColumn].every((column) => row[column] !== ""));
  } else {
    const imputations = {};
    [...config.features, config.targetColumn].forEach((column) => {
      const values = rows.map((row) => row[column]).filter((value) => value !== "");
      if (!values.length) return;
      imputations[column] = columnTypes[column] === "numeric" ? average(values.map(Number)) : mode(values);
    });

    rows = rows.map((row) => {
      const next = { ...row };
      Object.entries(imputations).forEach(([column, fillValue]) => {
        if (next[column] === "") next[column] = String(fillValue);
      });
      return next;
    });

  }

  if (!rows.length) return { ok: false, message: "No rows remain after preprocessing." };

  const artifacts = { encoders: {}, scalers: {}, imputations: {} };
  if (config.missingStrategy === "mean_mode") {
    [...config.features, config.targetColumn].forEach((column) => {
      const values = rawRows.map((row) => row[column]).filter((value) => value !== "");
      if (!values.length) return;
      artifacts.imputations[column] = columnTypes[column] === "numeric" ? average(values.map(Number)) : mode(values);
    });
  }
  const featureNames = [];
  const processedBase = rows.map(() => ({}));

  config.features.forEach((feature) => {
    if (columnTypes[feature] === "numeric") {
      const numericValues = rows.map((row) => Number(row[feature]));
      let transformedValues = [...numericValues];

      if (config.scalingStrategy === "standardize") {
        const mean = average(numericValues);
        const std = sampleStd(numericValues) || 1;
        artifacts.scalers[feature] = { type: "standardize", mean, std };
        transformedValues = numericValues.map((value) => (value - mean) / std);
      } else if (config.scalingStrategy === "normalize") {
        const min = Math.min(...numericValues);
        const max = Math.max(...numericValues);
        const range = max - min || 1;
        artifacts.scalers[feature] = { type: "normalize", min, max };
        transformedValues = numericValues.map((value) => (value - min) / range);
      }

      transformedValues.forEach((value, index) => {
        processedBase[index][feature] = value;
      });
      featureNames.push(feature);
    } else if (config.encodingStrategy === "label") {
      const categories = [...new Set(rows.map((row) => row[feature]))];
      const map = {};
      categories.forEach((category, index) => {
        map[category] = index;
      });
      artifacts.encoders[feature] = { type: "label", map };
      rows.forEach((row, index) => {
        processedBase[index][feature] = map[row[feature]];
      });
      featureNames.push(feature);
    } else if (config.encodingStrategy === "onehot") {
      const categories = [...new Set(rows.map((row) => row[feature]))];
      artifacts.encoders[feature] = { type: "onehot", categories };
      categories.forEach((category) => {
        const derivedName = `${feature}_${category}`;
        featureNames.push(derivedName);
        rows.forEach((row, index) => {
          processedBase[index][derivedName] = row[feature] === category ? 1 : 0;
        });
      });
    }
  });

  if (!featureNames.length) {
    return { ok: false, message: "No processed features were generated. Enable encoding or select at least one numeric feature." };
  }

  const targetValues = rows.map((row) => Number(row[config.targetColumn]));
  const processedRows = processedBase.map((item, index) => ({
    ...item,
    [config.targetColumn]: targetValues[index]
  }));

  return { ok: true, rows: processedRows, featureNames, targetValues, artifacts };
}

function renderBeforeAfterComparison(result) {
  els.beforeAfterComparison.innerHTML = [
    `<div class="mini-note"><strong>Before:</strong> ${state.rawRows.length} rows x ${state.columns.length} columns</div>`,
    `<div class="mini-note"><strong>After:</strong> ${result.rows.length} rows x ${result.featureNames.length + 1} columns</div>`,
    `<p><strong>Selected features:</strong> ${state.featureColumns.join(", ")}</p>`,
    `<p><strong>Missing handling:</strong> ${humanizeMissingStrategy(els.missingStrategy.value)}</p>`,
    `<p><strong>Encoding:</strong> ${humanizeEncoding(els.encodingStrategy.value)}</p>`,
    `<p><strong>Scaling:</strong> ${humanizeScaling(els.scalingStrategy.value)}</p>`
  ].join("");
}

function populateEdaFeatureSelector() {
  els.edaFeature.innerHTML = state.processedFeatureNames.map((feature) => `<option value="${feature}">${feature}</option>`).join("");
}

function renderEda() {
  if (!state.processedRows.length) return;
  const feature = els.edaFeature.value || state.processedFeatureNames[0];
  if (!feature) return;
  const xValues = state.processedRows.map((row) => Number(row[feature]));
  const yValues = state.targetValues;
  const corr = pearsonCorrelation(xValues, yValues);

  els.histogramChart.innerHTML = renderHistogramSvg(xValues, feature);
  els.scatterChart.innerHTML = renderScatterSvg(xValues, yValues, feature, state.targetColumn);
  els.edaNarrative.textContent = `${feature} has a correlation of ${corr.toFixed(3)} with ${state.targetColumn}. Values closer to 1 or -1 indicate a stronger linear relationship.`;
  els.correlationMatrix.innerHTML = renderCorrelationTable();
}

function renderCorrelationTable() {
  const numericColumns = [...state.processedFeatureNames, state.targetColumn];
  let html = '<div class="table-wrap"><table><thead><tr><th>Feature</th>';
  numericColumns.forEach((column) => {
    html += `<th>${column}</th>`;
  });
  html += "</tr></thead><tbody>";

  numericColumns.forEach((columnA) => {
    html += `<tr><th>${columnA}</th>`;
    const a = state.processedRows.map((row) => Number(row[columnA]));
    numericColumns.forEach((columnB) => {
      const b = state.processedRows.map((row) => Number(row[columnB]));
      const value = pearsonCorrelation(a, b);
      const alpha = Math.min(0.9, Math.abs(value));
      html += `<td style="background: rgba(201, 107, 51, ${alpha * 0.35});">${value.toFixed(3)}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody></table></div>";
  return html;
}

function trainModel() {
  if (!state.processedRows.length) {
    alert("Run preprocessing first.");
    return;
  }

  if (state.processedRows.length < 2) {
    alert("At least two processed rows are required for training and validation.");
    return;
  }

  const splitRatio = Number(els.splitRatio.value) / 100;
  const learningRate = Number(els.learningRate.value);
  const iterations = Number(els.iterations.value);
  const k = Number(els.kFolds.value);

  const dataset = state.processedRows.map((row) => ({
    x: state.processedFeatureNames.map((feature) => Number(row[feature])),
    y: Number(row[state.targetColumn])
  }));

  const trainSize = Math.max(1, Math.floor(dataset.length * splitRatio));
  const trainSet = dataset.slice(0, trainSize);
  const testSet = dataset.slice(trainSize);
  const trained = gradientDescent(trainSet, learningRate, iterations);
  const evaluationSet = testSet.length ? testSet : trainSet;
  const evaluationPredictions = evaluationSet.map((item) => predictFromWeights(trained.weights, item.x));
  const evaluationActuals = evaluationSet.map((item) => item.y);
  const metrics = calculateMetrics(evaluationActuals, evaluationPredictions);
  const cv = runKFoldValidation(dataset, k, learningRate, iterations);

  state.trainResult = {
    ...trained,
    trainSet,
    testSet,
    evaluationSet,
    evaluationPredictions,
    evaluationActuals,
    metrics,
    cv
  };

  renderLearningModule();
  renderTrainingOutputs();
  renderPredictionForm();
  renderStatus();
}

function gradientDescent(dataset, learningRate, iterations) {
  const featureCount = dataset[0].x.length;
  const weights = Array(featureCount + 1).fill(0);
  const costHistory = [];
  const gradientSnapshots = [];

  for (let iter = 0; iter < iterations; iter += 1) {
    const gradients = Array(featureCount + 1).fill(0);
    let costAccumulator = 0;

    dataset.forEach((item) => {
      const prediction = predictFromWeights(weights, item.x);
      const error = prediction - item.y;
      costAccumulator += error ** 2;
      gradients[0] += error;
      item.x.forEach((value, index) => {
        gradients[index + 1] += error * value;
      });
    });

    const m = dataset.length;
    for (let j = 0; j < weights.length; j += 1) {
      gradients[j] /= m;
      weights[j] -= learningRate * gradients[j];
    }

    costHistory.push(costAccumulator / (2 * dataset.length));
    if (iter < 5 || iter === iterations - 1) {
      gradientSnapshots.push({
        iteration: iter + 1,
        gradients: [...gradients],
        weights: [...weights],
        cost: costHistory[costHistory.length - 1]
      });
    }
  }

  return { weights, costHistory, gradientSnapshots };
}

function runKFoldValidation(dataset, k, learningRate, iterations) {
  const folds = Math.max(2, Math.min(k, dataset.length));
  const foldSize = Math.floor(dataset.length / folds);
  const results = [];

  for (let i = 0; i < folds; i += 1) {
    const start = i * foldSize;
    const end = i === folds - 1 ? dataset.length : start + foldSize;
    const validation = dataset.slice(start, end);
    const training = dataset.filter((_, index) => index < start || index >= end);
    const model = gradientDescent(training, learningRate, iterations);
    const predictions = validation.map((item) => predictFromWeights(model.weights, item.x));
    const actuals = validation.map((item) => item.y);
    results.push(calculateMetrics(actuals, predictions));
  }

  return {
    folds,
    results,
    averageMse: average(results.map((item) => item.mse)),
    averageMae: average(results.map((item) => item.mae)),
    averageR2: average(results.map((item) => item.r2))
  };
}

function renderLearningModule() {
  const sample = state.trainResult.trainSet[0];
  const prediction = predictFromWeights(state.trainResult.weights, sample.x);
  const error = prediction - sample.y;
  const terms = sample.x.map((value, index) => {
    const weight = state.trainResult.weights[index + 1];
    return `${formatNumber(weight)} x ${formatNumber(value)} = ${formatNumber(weight * value)}`;
  });

  els.workedSample.innerHTML = `
    <p><strong>Sample target:</strong> ${formatNumber(sample.y)}</p>
    <p><strong>Intercept contribution:</strong> ${formatNumber(state.trainResult.weights[0])}</p>
    <p><strong>Feature contributions:</strong></p>
    <ul>${terms.map((term) => `<li>${term}</li>`).join("")}</ul>
    <p><strong>Predicted value:</strong> ${formatNumber(prediction)}</p>
    <p><strong>Error:</strong> ${formatNumber(error)}</p>
  `;

  els.intermediateValues.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Iteration</th>
            <th>Cost</th>
            <th>Intercept Gradient</th>
            <th>Intercept</th>
          </tr>
        </thead>
        <tbody>
          ${state.trainResult.gradientSnapshots.map((snapshot) => `
            <tr>
              <td>${snapshot.iteration}</td>
              <td>${formatNumber(snapshot.cost)}</td>
              <td>${formatNumber(snapshot.gradients[0])}</td>
              <td>${formatNumber(snapshot.weights[0])}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderTrainingOutputs() {
  renderParameterTable();
  els.costChart.innerHTML = renderLineChartSvg(state.trainResult.costHistory, "Iteration", "Cost");
  els.predictionComparison.innerHTML = renderPredictionScatterSvg(
    state.trainResult.evaluationActuals,
    state.trainResult.evaluationPredictions
  );
  renderRegressionChart();
  renderMetrics();
  renderCvResults();
}

function renderParameterTable() {
  let html = '<div class="table-wrap"><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>';
  html += `<tr><td>Intercept</td><td>${formatNumber(state.trainResult.weights[0])}</td></tr>`;
  state.processedFeatureNames.forEach((feature, index) => {
    html += `<tr><td>${feature}</td><td>${formatNumber(state.trainResult.weights[index + 1])}</td></tr>`;
  });
  html += "</tbody></table></div>";
  els.parameterTable.innerHTML = html;
}

function renderRegressionChart() {
  if (state.processedFeatureNames.length !== 1) {
    els.regressionChart.innerHTML = `
      <p>Multiple linear regression is active because ${state.processedFeatureNames.length} processed features were selected.</p>
      <p>The model still learns a linear equation, but it exists in higher-dimensional space instead of a single 2D line.</p>
    `;
    return;
  }

  const feature = state.processedFeatureNames[0];
  const xValues = state.processedRows.map((row) => Number(row[feature]));
  const yValues = state.targetValues;
  els.regressionChart.innerHTML = renderRegressionLineSvg(
    xValues,
    yValues,
    state.trainResult.weights[0],
    state.trainResult.weights[1],
    feature,
    state.targetColumn
  );
}

function renderMetrics() {
  els.mseMetric.textContent = formatNumber(state.trainResult.metrics.mse);
  els.maeMetric.textContent = formatNumber(state.trainResult.metrics.mae);
  els.r2Metric.textContent = formatNumber(state.trainResult.metrics.r2);
}

function renderCvResults() {
  let html = `
    <p><strong>Average MSE:</strong> ${formatNumber(state.trainResult.cv.averageMse)}</p>
    <p><strong>Average MAE:</strong> ${formatNumber(state.trainResult.cv.averageMae)}</p>
    <p><strong>Average R²:</strong> ${formatNumber(state.trainResult.cv.averageR2)}</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Fold</th>
            <th>MSE</th>
            <th>MAE</th>
            <th>R²</th>
          </tr>
        </thead>
        <tbody>
  `;

  state.trainResult.cv.results.forEach((result, index) => {
    html += `
      <tr>
        <td>${index + 1}</td>
        <td>${formatNumber(result.mse)}</td>
        <td>${formatNumber(result.mae)}</td>
        <td>${formatNumber(result.r2)}</td>
      </tr>
    `;
  });

  html += "</tbody></table></div>";
  els.cvResults.innerHTML = html;
}

function renderPredictionForm() {
  const fields = state.featureColumns.map((feature) => {
    if (state.columnTypes[feature] === "numeric") {
      return `
        <label class="field">
          <span>${feature}</span>
          <input type="number" step="any" name="${feature}">
        </label>
      `;
    }

    const encoder = state.preprocessArtifacts.encoders[feature];
    const options = encoder
      ? (encoder.categories || Object.keys(encoder.map)).map((item) => `<option value="${item}">${item}</option>`).join("")
      : "";
    return `
      <label class="field">
        <span>${feature}</span>
        <select name="${feature}">${options}</select>
      </label>
    `;
  }).join("");

  els.predictionForm.innerHTML = `
    <form id="predictForm">
      ${fields}
      <button type="submit">Compute Prediction</button>
    </form>
  `;

  document.getElementById("predictForm").addEventListener("submit", handlePrediction);
}

function handlePrediction(event) {
  event.preventDefault();
  const formData = new FormData(event.target);
  const rawSample = {};
  state.featureColumns.forEach((feature) => {
    rawSample[feature] = formData.get(feature);
  });

  const transformed = transformNewSample(rawSample);
  const prediction = predictFromWeights(state.trainResult.weights, transformed.vector);
  const contributionList = transformed.vector.map((value, index) => {
    const weight = state.trainResult.weights[index + 1];
    return `<li>${state.processedFeatureNames[index]}: ${formatNumber(weight)} x ${formatNumber(value)} = ${formatNumber(weight * value)}</li>`;
  }).join("");

  els.predictionExplanation.innerHTML = `
    <p><strong>Hypothesis:</strong> y = theta0 + theta1x1 + ... + thetanxn</p>
    <p><strong>Intercept:</strong> ${formatNumber(state.trainResult.weights[0])}</p>
    <ul>${contributionList}</ul>
    <p><strong>Predicted value:</strong> ${formatNumber(prediction)}</p>
    <p>${transformed.notes.join(" ")}</p>
  `;
}

function transformNewSample(rawSample) {
  const vector = [];
  const notes = [];

  state.featureColumns.forEach((feature) => {
    let rawValue = rawSample[feature];
    if (state.columnTypes[feature] === "numeric") {
      if (rawValue === "" || rawValue === null) {
        rawValue = state.preprocessArtifacts.imputations[feature] ?? 0;
        notes.push(`${feature} was missing, so the training imputation value was used.`);
      }

      let value = Number(rawValue);
      if (Number.isNaN(value)) {
        value = Number(state.preprocessArtifacts.imputations[feature] ?? 0);
        notes.push(`${feature} could not be parsed, so the training imputation value was used.`);
      }

      const scaler = state.preprocessArtifacts.scalers[feature];
      if (scaler?.type === "standardize") {
        value = (value - scaler.mean) / scaler.std;
        notes.push(`${feature} was standardized using the training mean and standard deviation.`);
      } else if (scaler?.type === "normalize") {
        value = (value - scaler.min) / ((scaler.max - scaler.min) || 1);
        notes.push(`${feature} was normalized using the training minimum and maximum.`);
      }
      vector.push(value);
    } else {
      const encoder = state.preprocessArtifacts.encoders[feature];
      if ((rawValue === "" || rawValue === null) && state.preprocessArtifacts.imputations[feature] !== undefined) {
        rawValue = state.preprocessArtifacts.imputations[feature];
        notes.push(`${feature} was missing, so the training mode value was used.`);
      }
      if (encoder?.type === "label") {
        vector.push(encoder.map[rawValue] ?? 0);
        notes.push(`${feature} was label encoded before prediction.`);
      } else if (encoder?.type === "onehot") {
        encoder.categories.forEach((category) => {
          vector.push(rawValue === category ? 1 : 0);
        });
        notes.push(`${feature} was converted to one-hot columns before prediction.`);
      }
    }
  });

  return { vector, notes };
}

function renderFormulaPanel() {
  els.formulaPanel.innerHTML = `
    <div class="formula">Simple Regression: y-hat = theta0 + theta1x</div>
    <div class="formula">Multiple Regression: y-hat = theta0 + theta1x1 + theta2x2 + ... + thetanxn</div>
    <div class="formula">Error for one sample: error = y-hat - y</div>
    <div class="formula">Cost Function: J(theta) = (1 / 2m) * sum((y-hat - y)^2)</div>
    <div class="formula">Gradient Update: thetaj = thetaj - alpha * (1 / m) * sum((y-hat - y) * xj)</div>
  `;
}

function renderRegressionModes() {
  els.regressionModes.innerHTML = `
    <div class="mini-note">
      <strong>Simple Linear Regression</strong>
      <p>Uses one independent variable, so the fitted model can be drawn as a single straight line in 2D.</p>
    </div>
    <div class="mini-note">
      <strong>Multiple Linear Regression</strong>
      <p>Uses two or more independent variables. The hypothesis remains linear, but each extra feature adds another weighted term.</p>
    </div>
  `;
}

function resetTrainingOutputs() {
  resetModelOutputs();
}

function resetModelOutputs() {
  els.workedSample.innerHTML = "Train a model to generate a step-by-step numerical example.";
  els.intermediateValues.innerHTML = "Loss, gradient updates, and sample errors will be summarized after training.";
  els.parameterTable.innerHTML = "Train the model to inspect coefficients and intercept.";
  els.costChart.innerHTML = "Gradient descent cost values will appear here.";
  els.regressionChart.innerHTML = "If one feature is selected, the fitted line will be drawn here.";
  els.predictionComparison.innerHTML = "Train the model to compare predictions with actual values.";
  els.predictionForm.innerHTML = "Train the model to unlock a prediction form.";
  els.predictionExplanation.innerHTML = "The weighted sum and final prediction will be shown here.";
  els.mseMetric.textContent = "-";
  els.maeMetric.textContent = "-";
  els.r2Metric.textContent = "-";
  els.cvResults.innerHTML = "K-fold results will appear here after training.";
}

function predictFromWeights(weights, features) {
  return weights[0] + features.reduce((sum, value, index) => sum + weights[index + 1] * value, 0);
}

function calculateMetrics(actuals, predictions) {
  const errors = actuals.map((actual, index) => actual - predictions[index]);
  const mse = average(errors.map((error) => error ** 2));
  const mae = average(errors.map((error) => Math.abs(error)));
  const meanActual = average(actuals);
  const ssRes = errors.reduce((sum, error) => sum + error ** 2, 0);
  const ssTot = actuals.reduce((sum, actual) => sum + (actual - meanActual) ** 2, 0) || 1;
  const r2 = 1 - (ssRes / ssTot);
  return { mse, mae, r2 };
}

function renderTable(rows, columns) {
  let html = "<table><thead><tr>";
  columns.forEach((column) => {
    html += `<th>${column}</th>`;
  });
  html += "</tr></thead><tbody>";
  rows.forEach((row) => {
    html += "<tr>";
    columns.forEach((column) => {
      html += `<td>${formatCell(row[column])}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  return html;
}

function renderHistogramSvg(values, label) {
  const width = 520;
  const height = 260;
  const padding = 30;
  const bins = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const counts = Array(bins).fill(0);

  values.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor(((value - min) / span) * bins));
    counts[index] += 1;
  });

  const maxCount = Math.max(...counts, 1);
  const barWidth = (width - padding * 2) / bins;
  let bars = "";

  counts.forEach((count, index) => {
    const barHeight = (count / maxCount) * (height - padding * 2);
    const x = padding + index * barWidth;
    const y = height - padding - barHeight;
    bars += `<rect x="${x}" y="${y}" width="${barWidth - 8}" height="${barHeight}" fill="#c96b33" rx="8"></rect>`;
  });

  return `
    <div class="svg-wrap">
      <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        ${bars}
        <text x="${width / 2}" y="${height - 6}" text-anchor="middle" fill="#5f6b76">${label}</text>
      </svg>
    </div>
  `;
}

function renderScatterSvg(xValues, yValues, xLabel, yLabel) {
  const width = 520;
  const height = 280;
  const padding = 35;
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);

  const points = xValues.map((x, index) => {
    const y = yValues[index];
    const px = scale(x, xMin, xMax, padding, width - padding);
    const py = scale(y, yMin, yMax, height - padding, padding);
    return `<circle cx="${px}" cy="${py}" r="5" fill="#1f7a52" opacity="0.8"></circle>`;
  }).join("");

  return `
    <div class="svg-wrap">
      <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        ${points}
        <text x="${width / 2}" y="${height - 6}" text-anchor="middle" fill="#5f6b76">${xLabel}</text>
        <text x="18" y="${height / 2}" text-anchor="middle" fill="#5f6b76" transform="rotate(-90 18 ${height / 2})">${yLabel}</text>
      </svg>
    </div>
  `;
}

function renderLineChartSvg(values, xLabel, yLabel) {
  const width = 520;
  const height = 280;
  const padding = 35;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const points = values.map((value, index) => {
    const x = scale(index, 0, values.length - 1 || 1, padding, width - padding);
    const y = scale(value, min, max, height - padding, padding);
    return `${x},${y}`;
  }).join(" ");

  return `
    <div class="svg-wrap">
      <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        <polyline points="${points}" fill="none" stroke="#c96b33" stroke-width="3"></polyline>
        <text x="${width / 2}" y="${height - 6}" text-anchor="middle" fill="#5f6b76">${xLabel}</text>
        <text x="18" y="${height / 2}" text-anchor="middle" fill="#5f6b76" transform="rotate(-90 18 ${height / 2})">${yLabel}</text>
      </svg>
    </div>
  `;
}

function renderRegressionLineSvg(xValues, yValues, intercept, weight, xLabel, yLabel) {
  const width = 520;
  const height = 280;
  const padding = 35;
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const points = xValues.map((x, index) => {
    const y = yValues[index];
    const px = scale(x, xMin, xMax, padding, width - padding);
    const py = scale(y, yMin, yMax, height - padding, padding);
    return `<circle cx="${px}" cy="${py}" r="5" fill="#1f7a52" opacity="0.8"></circle>`;
  }).join("");

  const lineStartY = intercept + weight * xMin;
  const lineEndY = intercept + weight * xMax;
  const x1 = scale(xMin, xMin, xMax, padding, width - padding);
  const x2 = scale(xMax, xMin, xMax, padding, width - padding);
  const y1 = scale(lineStartY, yMin, yMax, height - padding, padding);
  const y2 = scale(lineEndY, yMin, yMax, height - padding, padding);

  return `
    <div class="svg-wrap">
      <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#7b6c59"></line>
        ${points}
        <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#c96b33" stroke-width="3"></line>
        <text x="${width / 2}" y="${height - 6}" text-anchor="middle" fill="#5f6b76">${xLabel}</text>
        <text x="18" y="${height / 2}" text-anchor="middle" fill="#5f6b76" transform="rotate(-90 18 ${height / 2})">${yLabel}</text>
      </svg>
    </div>
  `;
}

function renderPredictionScatterSvg(actuals, predictions) {
  return renderScatterSvg(actuals, predictions, "Actual", "Predicted");
}

function scale(value, oldMin, oldMax, newMin, newMax) {
  if (oldMax === oldMin) return (newMin + newMax) / 2;
  return newMin + ((value - oldMin) / (oldMax - oldMin)) * (newMax - newMin);
}

function pearsonCorrelation(a, b) {
  const meanA = average(a);
  const meanB = average(b);
  let numerator = 0;
  let denomA = 0;
  let denomB = 0;

  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    numerator += da * db;
    denomA += da ** 2;
    denomB += db ** 2;
  }

  return numerator / Math.sqrt((denomA || 1) * (denomB || 1));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleStd(values) {
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function mode(values) {
  const counts = {};
  let bestValue = values[0];
  let bestCount = 0;
  values.forEach((value) => {
    counts[value] = (counts[value] || 0) + 1;
    if (counts[value] > bestCount) {
      bestCount = counts[value];
      bestValue = value;
    }
  });
  return bestValue;
}

function humanizeMissingStrategy(value) {
  return value === "mean_mode" ? "Mean for numeric columns and mode for categorical columns" : "Drop incomplete rows";
}

function humanizeEncoding(value) {
  return value === "onehot" ? "One-hot encoding" : value === "label" ? "Label encoding" : "No encoding";
}

function humanizeScaling(value) {
  return value === "standardize" ? "Standardization" : value === "normalize" ? "Normalization" : "No scaling";
}

function formatCell(value) {
  if (typeof value === "number") return formatNumber(value);
  return value;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  return Number(value).toFixed(4);
}
