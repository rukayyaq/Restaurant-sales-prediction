// THIS MUST PRINT IN THE CONSOLE. IF IT DOESN'T, THE FILE IS NOT LOADING.
console.log("🚀 analytics.js has started executing!");

document.addEventListener('DOMContentLoaded', () => {
    console.log("📊 DOM loaded, fetching data...");

    const $ = id => document.getElementById(id);

    async function getData(url) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`Fetch failed for ${url}:`, err);
            document.body.insertAdjacentHTML('afterbegin', `<div class="alert alert-danger m-3">Error loading ${url}: ${err.message}</div>`);
            return null;
        }
    }

    function makeChart(id, type, data, options = {}) {
        const el = $(id);
        if (!el) return console.warn(`Element #${id} not found!`);
        if (typeof Chart === 'undefined') {
            console.error("Chart.js is not loaded! Check your internet connection or the script tag.");
            return;
        }
        
        new Chart(el, {
            type: type,
            data: data,
            options: Object.assign({
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }, options)
        });
    }

    async function initDashboard() {
        // 1. Summary
        const s = await getData('/api/analytics/summary');
        if (s) {
            $('kpi-sales').textContent = s.total_sales.toLocaleString();
            $('kpi-revenue').textContent = 'RM ' + s.total_revenue.toLocaleString();
            $('kpi-profit').textContent = 'RM ' + s.total_profit.toLocaleString();

            makeChart('chartTop', 'bar', {
                labels: s.top_dishes.labels,
                datasets: [{ label: 'Units', data: s.top_dishes.values, backgroundColor: '#6366f1' }]
            });
            makeChart('chartMeal', 'bar', {
                labels: s.meal_type.labels,
                datasets: [{ label: 'Avg Units', data: s.meal_type.values, backgroundColor: '#10b981' }]
            });
            makeChart('chartWeather', 'bar', {
                labels: s.weather.labels,
                datasets: [{ label: 'Avg Units', data: s.weather.values, backgroundColor: '#f59e0b' }]
            });
        }

        // 2. Revenue
        const r = await getData('/api/analytics/revenue');
        if (r) {
            makeChart('chartRevenue', 'line', {
                labels: r.monthly.map(m => 'M' + m.month),
                datasets: [
                    { label: 'Revenue', data: r.monthly.map(m => m.revenue), borderColor: '#10b981', tension: 0.3 },
                    { label: 'Profit', data: r.monthly.map(m => m.profit), borderColor: '#6366f1', tension: 0.3 }
                ]
            });
        }

        // 3. Heatmap
        const h = await getData('/api/analytics/heatmap');
        if (h) {
            const colors = ['#6366f1', '#ec4899', '#f59e0b'];
            makeChart('chartHeat', 'bar', {
                labels: h.rows,
                datasets: h.cols.map((col, i) => ({
                    label: col,
                    data: h.data.map(row => row[i]),
                    backgroundColor: colors[i % colors.length]
                }))
            }, { scales: { x: { stacked: true }, y: { stacked: true } } });
        }

        // 4. Impact
        const imp = await getData('/api/analytics/impact');
        if (imp) {
            $('impactBox').innerHTML = `
                <div class="alert alert-success py-2">Promotion Lift: <strong>${imp.promotion_lift_pct}%</strong></div>
                <ul class="mt-3">${Object.entries(imp.weather_effect).map(([k,v]) => `<li>${k}: <strong>${v}</strong> units</li>`).join('')}</ul>
            `;
        }

        // 5. Margins
        const m = await getData('/api/analytics/margins');
        if (m) {
            $('marginTable').innerHTML = `
                <thead><tr><th>Item</th><th>Avg Price</th><th>Avg Cost</th><th>Margin %</th><th>Units</th></tr></thead>
                <tbody>${m.map(row => `
                    <tr>
                        <td>${row.menu_item_name}</td>
                        <td>RM ${row.avg_price}</td>
                        <td>RM ${row.avg_cost}</td>
                        <td><span class="badge ${row.avg_margin_pct > 50 ? 'bg-success' : 'bg-warning'}">${row.avg_margin_pct}%</span></td>
                        <td>${row.units}</td>
                    </tr>`).join('')}
                </tbody>`;
        }
        console.log("🎉 Dashboard fully loaded.");
    }

    initDashboard();
});