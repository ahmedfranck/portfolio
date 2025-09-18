// Sales Performance Dashboard JS
// Dataset is loaded from data.js and exposed as `datasetData`.

// Assign dataset from global datasetData or default to empty array
let dataset = typeof datasetData !== 'undefined' ? datasetData : [];

// Helper function: group by key and sum values
function groupBy(data, key, valueKey) {
    const result = {};
    data.forEach(item => {
        const group = item[key];
        if (!result[group]) {
            result[group] = 0;
        }
        result[group] += item[valueKey];
    });
    return result;
}

// Helper function: filter dataset based on filter values
function filterDataset() {
    const start = document.getElementById('startDate').value;
    const end = document.getElementById('endDate').value;
    const regions = Array.from(document.getElementById('regionSelect').selectedOptions).map(o => o.value);
    const categories = Array.from(document.getElementById('categorySelect').selectedOptions).map(o => o.value);
    const reps = Array.from(document.getElementById('repSelect').selectedOptions).map(o => o.value);

    return dataset.filter(item => {
        const itemDate = item.order_date;
        const dateInRange = (!start || itemDate >= start) && (!end || itemDate <= end);
        const regionMatch = (regions.length === 0 || regions.includes(item.region));
        const categoryMatch = (categories.length === 0 || categories.includes(item.category));
        const repMatch = (reps.length === 0 || reps.includes(item.sales_rep));
        return dateInRange && regionMatch && categoryMatch && repMatch;
    });
}

// Populate filter options based on dataset values
function initFilters() {
    const regionSelect = document.getElementById('regionSelect');
    const categorySelect = document.getElementById('categorySelect');
    const repSelect = document.getElementById('repSelect');

    const regions = [...new Set(dataset.map(d => d.region))];
    const categories = [...new Set(dataset.map(d => d.category))];
    const reps = [...new Set(dataset.map(d => d.sales_rep))];

    regions.forEach(region => {
        const opt = document.createElement('option');
        opt.value = region;
        opt.textContent = region;
        regionSelect.appendChild(opt);
    });
    categories.forEach(cat => {
        const opt = document.createElement('option');
        opt.value = cat;
        opt.textContent = cat;
        categorySelect.appendChild(opt);
    });
    reps.forEach(rep => {
        const opt = document.createElement('option');
        opt.value = rep;
        opt.textContent = rep;
        repSelect.appendChild(opt);
    });
}

// Update charts based on filtered data
function updateDashboard() {
    const filteredData = filterDataset();
    if (filteredData.length === 0) {
        // handle empty data case
        Plotly.newPlot('salesTrend', [{x: [], y: [], type: 'scatter'}], {title: 'No data for selected filters'});
        Plotly.newPlot('topProducts', [{x: [], y: [], type: 'bar'}], {title: 'No data for selected filters'});
        Plotly.newPlot('categoryBreakdown', [{labels: [], values: [], type: 'pie'}], {title: 'No data for selected filters'});
        Plotly.newPlot('regionPerformance', [{x: [], y: [], type: 'bar'}], {title: 'No data for selected filters'});
        return;
    }
    // Sales Trend: aggregate by date
    const dateMap = {};
    filteredData.forEach(item => {
        const date = item.order_date;
        if (!dateMap[date]) dateMap[date] = 0;
        dateMap[date] += item.sales;
    });
    const dates = Object.keys(dateMap).sort();
    const salesTrend = dates.map(d => dateMap[d]);
    const trendTrace = {
        x: dates,
        y: salesTrend,
        mode: 'lines+markers',
        line: {color: '#007ACC'},
        name: 'Sales'
    };
    const trendLayout = {
        title: 'Sales Trend',
        xaxis: {title: 'Date'},
        yaxis: {title: 'Sales'},
        margin: {l: 50, r: 20, t: 40, b: 50},
        hovermode: 'closest'
    };
    Plotly.newPlot('salesTrend', [trendTrace], trendLayout);

    // Top Products by Sales
    const productSales = groupBy(filteredData, 'product', 'sales');
    const topProducts = Object.entries(productSales).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const prodNames = topProducts.map(item => item[0]);
    const prodSales = topProducts.map(item => item[1]);
    const topTrace = {
        x: prodNames,
        y: prodSales,
        type: 'bar',
        marker: {color: '#22A699'},
    };
    const topLayout = {
        title: 'Top Products by Sales',
        xaxis: {title: 'Product'},
        yaxis: {title: 'Sales'},
        margin: {l: 50, r: 20, t: 40, b: 80},
    };
    Plotly.newPlot('topProducts', [topTrace], topLayout);

    // Category Breakdown (Pie)
    const categorySales = groupBy(filteredData, 'category', 'sales');
    const catLabels = Object.keys(categorySales);
    const catValues = Object.values(categorySales);
    const categoryTrace = {
        labels: catLabels,
        values: catValues,
        type: 'pie',
        hole: 0.4,
    };
    const categoryLayout = {
        title: 'Revenue Breakdown by Category',
        showlegend: true,
        margin: {l: 20, r: 20, t: 40, b: 20},
    };
    Plotly.newPlot('categoryBreakdown', [categoryTrace], categoryLayout);

    // Regional Performance (Bar)
    const regionSales = groupBy(filteredData, 'region', 'sales');
    const regNames = Object.keys(regionSales);
    const regValues = Object.values(regionSales);
    const regionTrace = {
        x: regNames,
        y: regValues,
        type: 'bar',
        marker: {color: '#EF6C00'},
    };
    const regionLayout = {
        title: 'Performance by Region',
        xaxis: {title: 'Region'},
        yaxis: {title: 'Sales'},
        margin: {l: 50, r: 20, t: 40, b: 50},
    };
    Plotly.newPlot('regionPerformance', [regionTrace], regionLayout);
}

// Initialize filters and charts once the page is loaded.
document.addEventListener('DOMContentLoaded', function () {
    if (dataset && dataset.length > 0) {
        initFilters();
        const dates = dataset.map(d => d.order_date).sort();
        document.getElementById('startDate').value = dates[0];
        document.getElementById('endDate').value = dates[dates.length - 1];
        updateDashboard();
    }
});

// Attach change listeners to filters
document.addEventListener('change', function (event) {
    const targetIds = ['startDate', 'endDate', 'regionSelect', 'categorySelect', 'repSelect'];
    if (targetIds.includes(event.target.id)) {
        updateDashboard();
    }
});