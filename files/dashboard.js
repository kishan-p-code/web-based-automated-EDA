// Dashboard charts — fetches aggregated data from /api/chart-data and renders
// with Chart.js, styled to match the garage-dashboard theme (dark surfaces,
// amber accent).

const COLORS = {
    amber: '#f5a623',
    green: '#5ec98f',
    red: '#e8607a',
    blue: '#5e9ec9',
    purple: '#a685d6',
    text: '#9aa0ab',
    grid: '#343943',
};

const PALETTE = [COLORS.amber, COLORS.blue, COLORS.green, COLORS.purple, COLORS.red, '#e8c45e'];

Chart.defaults.color = COLORS.text;
Chart.defaults.font.family = "'Inter', sans-serif";

async function loadCharts() {
    const res = await fetch('/api/chart-data');
    const data = await res.json();

    renderBarChart('chartBrand', data.by_brand, 'Vehicles');
    renderDoughnutChart('chartFuel', data.by_fuel);
    renderDoughnutChart('chartStatus', data.by_status, {
        Available: COLORS.green,
        Reserved: COLORS.amber,
        Sold: COLORS.red,
    });
    renderLineChart('chartYear', data.by_year);
}

function renderBarChart(canvasId, dataObj, label) {
    const ctx = document.getElementById(canvasId);
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(dataObj),
            datasets: [{
                label,
                data: Object.values(dataObj),
                backgroundColor: COLORS.amber,
                borderRadius: 4,
                maxBarThickness: 36,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: COLORS.grid }, beginAtZero: true, ticks: { precision: 0 } },
            },
        },
    });
}

function renderDoughnutChart(canvasId, dataObj, colorMap) {
    const ctx = document.getElementById(canvasId);
    const labels = Object.keys(dataObj);
    const colors = colorMap
        ? labels.map(l => colorMap[l] || COLORS.text)
        : labels.map((_, i) => PALETTE[i % PALETTE.length]);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: Object.values(dataObj),
                backgroundColor: colors,
                borderColor: '#1e2128',
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 12, padding: 12, font: { size: 12 } },
                },
            },
        },
    });
}

function renderLineChart(canvasId, dataObj) {
    const ctx = document.getElementById(canvasId);
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Object.keys(dataObj),
            datasets: [{
                label: 'Total Value ($)',
                data: Object.values(dataObj),
                borderColor: COLORS.amber,
                backgroundColor: 'rgba(245, 166, 35, 0.1)',
                fill: true,
                tension: 0.3,
                pointBackgroundColor: COLORS.amber,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: COLORS.grid }, ticks: { callback: v => '$' + v.toLocaleString() } },
            },
        },
    });
}

loadCharts();
