/**
 * AutoValuate AI - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    initYearDropdown();
    initPriceHelper();
    initPresets();
    initTabSwitcher();
    initFeatureImportanceChart();
    initFormHandler();
});

// Fallback presets if backend API is not immediately available
const DEFAULT_PRESETS = [
    {
        car_name: "Maruti Swift VXi",
        present_price: 5.90,
        year: 2018,
        kms_driven: 42000,
        fuel_type: "Petrol",
        seller_type: "Dealer",
        transmission: "Manual",
        owner: 0
    },
    {
        car_name: "Hyundai i20 Asta",
        present_price: 7.50,
        year: 2019,
        kms_driven: 35000,
        fuel_type: "Petrol",
        seller_type: "Dealer",
        transmission: "Manual",
        owner: 0
    },
    {
        car_name: "Honda City VX",
        present_price: 11.20,
        year: 2016,
        kms_driven: 58000,
        fuel_type: "Petrol",
        seller_type: "Dealer",
        transmission: "Automatic",
        owner: 1
    },
    {
        car_name: "Toyota Fortuner 4x2",
        present_price: 32.50,
        year: 2018,
        kms_driven: 65000,
        fuel_type: "Diesel",
        seller_type: "Dealer",
        transmission: "Automatic",
        owner: 0
    },
    {
        car_name: "Maruti Alto 800",
        present_price: 3.80,
        year: 2014,
        kms_driven: 62000,
        fuel_type: "Petrol",
        seller_type: "Individual",
        transmission: "Manual",
        owner: 1
    }
];

/**
 * 1. Initialize Year Dropdown from 2024 down to 2005
 */
function initYearDropdown() {
    const yearSelect = document.getElementById('year');
    if (!yearSelect) return;

    const currentYear = 2024;
    for (let yr = currentYear; yr >= 2005; yr--) {
        const opt = document.createElement('option');
        opt.value = yr;
        opt.textContent = yr;
        if (yr === 2019) opt.selected = true;
        yearSelect.appendChild(opt);
    }
}

/**
 * 2. Realtime INR formatting helper for Showroom Price
 */
function initPriceHelper() {
    const priceInput = document.getElementById('presentPrice');
    const helper = document.getElementById('inrHelper');
    if (!priceInput || !helper) return;

    function updateHelper() {
        const val = parseFloat(priceInput.value);
        if (!isNaN(val) && val > 0) {
            const inr = Math.round(val * 100000);
            helper.textContent = `₹ ${inr.toLocaleString('en-IN')}`;
        } else {
            helper.textContent = '₹ 0';
        }
    }

    priceInput.addEventListener('input', updateHelper);
    updateHelper();
}

/**
 * 3. Populate Quick Presets Chips
 */
async function initPresets() {
    const container = document.getElementById('presetChips');
    if (!container) return;

    let presets = DEFAULT_PRESETS;
    try {
        const res = await fetch('/api/presets');
        if (res.ok) {
            const data = await res.json();
            if (data.presets && data.presets.length > 0) {
                presets = data.presets;
            }
        }
    } catch (e) {
        console.log('Using default local presets.');
    }

    container.innerHTML = '';
    presets.forEach(p => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'preset-chip';
        chip.innerHTML = `🚘 ${p.car_name} (${p.year})`;
        chip.addEventListener('click', () => applyPreset(p));
        container.appendChild(chip);
    });
}

function applyPreset(p) {
    document.getElementById('carName').value = p.car_name;
    document.getElementById('presentPrice').value = p.present_price;
    document.getElementById('year').value = p.year;
    document.getElementById('kmsDriven').value = p.kms_driven;
    document.getElementById('owner').value = p.owner;

    // Fuel Type
    const fuelRadio = document.querySelector(`input[name="fuelType"][value="${p.fuel_type}"]`);
    if (fuelRadio) fuelRadio.checked = true;

    // Seller Type
    const sellerRadio = document.querySelector(`input[name="sellerType"][value="${p.seller_type}"]`);
    if (sellerRadio) sellerRadio.checked = true;

    // Transmission
    const transRadio = document.querySelector(`input[name="transmission"][value="${p.transmission}"]`);
    if (transRadio) transRadio.checked = true;

    // Update price helper
    const helper = document.getElementById('inrHelper');
    if (helper) {
        helper.textContent = `₹ ${(p.present_price * 100000).toLocaleString('en-IN')}`;
    }

    // Automatically trigger estimation
    submitForm();
}

/**
 * 4. Form Submission and Prediction Rendering
 */
function initFormHandler() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm();
    });
}

async function submitForm() {
    const btn = document.getElementById('predictBtn');
    const spinner = document.getElementById('spinner');
    const btnText = btn.querySelector('.btn-text');

    const carName = document.getElementById('carName').value || 'Used Car';
    const presentPrice = parseFloat(document.getElementById('presentPrice').value);
    const year = parseInt(document.getElementById('year').value);
    const kmsDriven = parseInt(document.getElementById('kmsDriven').value);
    const owner = parseInt(document.getElementById('owner').value);

    const fuelType = document.querySelector('input[name="fuelType"]:checked')?.value || 'Petrol';
    const sellerType = document.querySelector('input[name="sellerType"]:checked')?.value || 'Dealer';
    const transmission = document.querySelector('input[name="transmission"]:checked')?.value || 'Manual';

    if (isNaN(presentPrice) || presentPrice <= 0) {
        alert('Please enter a valid showroom price in ₹ Lakhs.');
        return;
    }

    const payload = {
        car_name: carName,
        present_price: presentPrice,
        year: year,
        kms_driven: kmsDriven,
        fuel_type: fuelType,
        seller_type: sellerType,
        transmission: transmission,
        owner: owner
    };

    // UI Loading state
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Calculating Fair Price...';
    btn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.message || 'Prediction failed');
        }

        const data = await response.json();
        renderPrediction(data);
    } catch (err) {
        console.error('Prediction Error:', err);
        // Fallback local estimation if backend is offline
        renderFallbackPrediction(payload);
    } finally {
        spinner.style.display = 'none';
        btnText.textContent = '⚡ Calculate Estimated Selling Price';
        btn.disabled = false;
    }
}

function renderPrediction(data) {
    document.getElementById('placeholderView').style.display = 'none';
    document.getElementById('predictionView').style.display = 'flex';

    document.getElementById('resCarName').textContent = data.car_name;
    document.getElementById('resAgePill').textContent = `${data.car_age_years} Years Old`;
    document.getElementById('resPriceLakhs').textContent = data.predicted_price_lakhs.toFixed(2);
    document.getElementById('resPriceINR').textContent = data.predicted_price_inr;

    document.getElementById('resShowroom').textContent = `₹${data.present_price_lakhs.toFixed(2)} L`;
    document.getElementById('resLoss').textContent = `₹${data.depreciation_lakhs.toFixed(2)} L`;
    document.getElementById('resDepPct').textContent = `${data.depreciation_pct}%`;

    const retainedPct = Math.max(0, 100 - data.depreciation_pct);
    document.getElementById('resRetainedBar').style.width = `${retainedPct}%`;
    document.getElementById('resRetainedLabel').textContent = `${retainedPct.toFixed(1)}%`;

    document.getElementById('resTierText').textContent = data.valuation_tier;

    const tierBadge = document.getElementById('tierBadge');
    tierBadge.textContent = data.tier_badge;
    tierBadge.className = 'badge-tier';
    if (data.depreciation_pct < 35) {
        tierBadge.classList.add('excellent');
    } else if (data.depreciation_pct < 60) {
        tierBadge.classList.add('fair');
    }
}

function renderFallbackPrediction(payload) {
    // Graceful offline estimator
    const age = 2024 - payload.year;
    let depRate = 0.12 + (age * 0.07) + (payload.kms_driven / 200000) * 0.1;
    if (payload.seller_type === 'Individual') depRate += 0.03;
    if (payload.transmission === 'Manual') depRate += 0.02;
    depRate = Math.min(0.85, Math.max(0.15, depRate));

    const estLakhs = Math.max(0.2, payload.present_price * (1 - depRate));
    const loss = payload.present_price - estLakhs;
    const depPct = (loss / payload.present_price) * 100;

    renderPrediction({
        car_name: payload.car_name,
        car_age_years: age,
        predicted_price_lakhs: estLakhs,
        predicted_price_inr: `₹${Math.round(estLakhs * 100000).toLocaleString('en-IN')}`,
        present_price_lakhs: payload.present_price,
        depreciation_lakhs: loss,
        depreciation_pct: Math.round(depPct * 10) / 10,
        valuation_tier: depPct < 35 ? 'High Value Retention' : 'Moderate Depreciation',
        tier_badge: depPct < 35 ? 'Excellent' : 'Fair Market'
    });
}

/**
 * 5. Tab Switcher
 */
function initTabSwitcher() {
    const buttons = document.querySelectorAll('.tab-btn');
    const panes = document.querySelectorAll('.tab-pane');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            buttons.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
        });
    });
}

/**
 * 6. Interactive Feature Importance Chart with Chart.js
 */
async function initFeatureImportanceChart() {
    const canvas = document.getElementById('featureChart');
    if (!canvas) return;

    let featureData = {
        'Present Price': 0.865,
        'Car Age': 0.082,
        'Kms Driven': 0.034,
        'Fuel (Diesel)': 0.009,
        'Transmission (Manual)': 0.005,
        'Seller (Individual)': 0.003,
        'Owner': 0.002
    };

    try {
        const res = await fetch('/api/metrics');
        if (res.ok) {
            const data = await res.json();
            if (data.feature_importances) {
                const raw = data.feature_importances;
                const formatted = {};
                for (const [k, v] of Object.entries(raw)) {
                    let label = k.replace(/_/g, ' ');
                    if (label === 'Present Price') label = 'Present / Showroom Price';
                    formatted[label] = v;
                }
                featureData = formatted;
            }
        }
    } catch (err) {
        console.log('Using default feature importance data.');
    }

    // Sort descending
    const sortedEntries = Object.entries(featureData).sort((a, b) => b[1] - a[1]);
    const labels = sortedEntries.map(e => e[0]);
    const values = sortedEntries.map(e => Math.round(e[1] * 1000) / 10); // in percentage

    new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Predictive Importance (%)',
                data: values,
                backgroundColor: [
                    '#3b82f6',
                    '#60a5fa',
                    '#93c5fd',
                    '#10b981',
                    '#34d399',
                    '#6ee7b7',
                    '#a7f3d0'
                ],
                borderRadius: 6,
                borderWidth: 0
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` ${context.parsed.x}% relative importance`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#9ca3af',
                        callback: val => `${val}%`
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f3f4f6', font: { weight: '600' } }
                }
            }
        }
    });
}
