let priceChart;
let tickerCatalog = [];

const form = document.getElementById("prediction-form");
const tickerInput = document.getElementById("ticker-input");
const periodSelect = document.getElementById("period-select");
const targetDateInput = document.getElementById("target-date-input");
const statusText = document.getElementById("status-text");
const tickerDropdown = document.getElementById("ticker-dropdown");
const tickerResults = document.getElementById("ticker-results");
const toggleTickerListButton = document.getElementById("toggle-ticker-list");

async function loadTickerOptions() {
  try {
    const response = await fetch("/static/tickers.json");
    if (!response.ok) {
      return;
    }

    tickerCatalog = await response.json();
    renderTickerList(tickerCatalog);
  } catch (error) {
    console.error("Unable to load ticker options:", error);
  }
}

async function fetchPrediction(ticker, period, targetDate = "") {
  statusText.textContent = `Loading ${ticker} data and training the model...`;

  try {
    const params = new URLSearchParams({ ticker, period });
    if (targetDate) {
      params.set("target_date", targetDate);
    }

    const response = await fetch(`/api/predict?${params.toString()}`);
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Unable to fetch prediction data.");
    }

    renderDashboard(payload);
    const sourceNote = payload.dataSource === "Yahoo Finance" ? "live Yahoo Finance data" : "demo fallback data";
    statusText.textContent = `Showing ${payload.ticker} using ${payload.period} of market history from ${sourceNote}.`;
  } catch (error) {
    statusText.textContent = error.message;
  }
}

function renderDashboard(payload) {
  document.getElementById("company-name").textContent = payload.companyName || payload.ticker;
  document.getElementById("ticker-pill").textContent = payload.ticker;
  document.getElementById("last-close").textContent = formatCurrency(payload.summary.lastClose);
  document.getElementById("pred-close").textContent = formatCurrency(payload.summary.predictedNextClose);
  document.getElementById("pred-change").textContent = `${payload.summary.predictedChangePct}%`;
  document.getElementById("target-date-label").textContent = payload.forecastMeta.targetTradingDate;
  document.getElementById("rsi-value").textContent = payload.summary.rsi14;
  document.getElementById("mae-value").textContent = payload.metrics.mae;
  document.getElementById("r2-value").textContent = payload.metrics.r2;
  document.getElementById("train-size").textContent = payload.metrics.trainSize;
  document.getElementById("test-size").textContent = payload.metrics.testSize;
  document.getElementById("prediction-band-value").textContent =
    `${payload.predictionPoint.date}: ${formatCurrency(payload.predictionPoint.lowerBound)} to ${formatCurrency(payload.predictionPoint.upperBound)}`;
  document.getElementById("forecast-window").textContent =
    `Forecasting ${payload.forecastMeta.sessionsAhead} trading session(s) ahead`;
  document.getElementById("last-market-date").textContent = payload.forecastMeta.lastMarketDate;
  document.getElementById("prediction-date").textContent = payload.forecastMeta.targetTradingDate;
  document.getElementById("sessions-ahead").textContent = payload.forecastMeta.sessionsAhead;
  document.getElementById("source-chip").textContent = `Source: ${payload.dataSource}`;

  const insightList = document.getElementById("insight-list");
  insightList.innerHTML = "";
  payload.insights.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    insightList.appendChild(li);
  });

  renderChart(payload);
}

function renderChart(payload) {
  const labels = payload.chartData.map((point) => point.date);
  const closeData = payload.chartData.map((point) => point.close);
  const ma7Data = payload.chartData.map((point) => point.ma7);
  const ma21Data = payload.chartData.map((point) => point.ma21);
  const predictionLabel = payload.predictionPoint.date;

  labels.push(predictionLabel);
  closeData.push(null);
  ma7Data.push(null);
  ma21Data.push(null);

  const predictionData = new Array(labels.length - 1).fill(null);
  predictionData.push(payload.predictionPoint.predictedClose);

  const ctx = document.getElementById("price-chart");
  if (priceChart) {
    priceChart.destroy();
  }

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close",
          data: closeData,
          borderColor: "#0f172a",
          backgroundColor: "rgba(15, 23, 42, 0.08)",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 0,
          fill: true,
        },
        {
          label: "7-Day Moving Average",
          data: ma7Data,
          borderColor: "#f97316",
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0,
        },
        {
          label: "21-Day Moving Average",
          data: ma21Data,
          borderColor: "#0891b2",
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0,
        },
        {
          label: `Predicted close (${payload.predictionPoint.date})`,
          data: predictionData,
          borderColor: "#16a34a",
          backgroundColor: "#16a34a",
          borderDash: [6, 6],
          borderWidth: 3,
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: "#172033",
          padding: 16,
          displayColors: false,
          cornerRadius: 12,
          titleFont: {
            size: 16,
            weight: "700",
          },
          bodyFont: {
            size: 14,
            weight: "600",
          },
          bodySpacing: 6,
          caretPadding: 10,
          callbacks: {
            title(items) {
              return items[0]?.label || "";
            },
            label(context) {
              if (context.parsed.y === null) {
                return "";
              }
              return `${context.dataset.label} ${formatCurrency(context.parsed.y)}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: {
            display: false,
          },
        },
        y: {
          ticks: {
            callback(value) {
              return formatCurrency(value);
            },
          },
        },
      },
    },
  });
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function renderTickerList(items) {
  tickerResults.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "ticker-empty";
    empty.textContent = "No tickers found.";
    tickerResults.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "ticker-option";
    row.innerHTML = `<strong>${item.symbol}</strong><span>${item.name}</span>`;
    row.addEventListener("click", () => {
      tickerInput.value = item.symbol;
      tickerDropdown.classList.add("hidden");
    });
    tickerResults.appendChild(row);
  });
}

function filterTickerList() {
  const query = tickerInput.value.trim().toLowerCase();
  const filtered = tickerCatalog.filter((item) => {
    return item.symbol.toLowerCase().includes(query) || item.name.toLowerCase().includes(query);
  });

  renderTickerList(filtered);
  tickerDropdown.classList.remove("hidden");
}

function defaultTargetDate() {
  const date = new Date();
  date.setDate(date.getDate() + 7);
  return date.toISOString().split("T")[0];
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  fetchPrediction(tickerInput.value.trim() || "RELIANCE.NS", periodSelect.value, targetDateInput.value);
});

tickerInput.addEventListener("focus", () => {
  renderTickerList(tickerCatalog);
  tickerDropdown.classList.remove("hidden");
});

tickerInput.addEventListener("input", () => {
  filterTickerList();
});

toggleTickerListButton.addEventListener("click", () => {
  const isHidden = tickerDropdown.classList.contains("hidden");

  if (isHidden) {
    renderTickerList(tickerCatalog);
    tickerDropdown.classList.remove("hidden");
  } else {
    tickerDropdown.classList.add("hidden");
  }
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".ticker-picker")) {
    tickerDropdown.classList.add("hidden");
  }
});

loadTickerOptions().then(() => {
  targetDateInput.value = defaultTargetDate();
  fetchPrediction(tickerInput.value, periodSelect.value, targetDateInput.value);
});
