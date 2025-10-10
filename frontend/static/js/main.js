// Main JavaScript for IDS Model Interface

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Auto-dismiss only alerts explicitly marked
    const alerts = document.querySelectorAll('.alert[data-auto-dismiss="true"]');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Add loading state to forms
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                submitBtn.disabled = true;
            }
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add animation to model cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe model cards
    document.querySelectorAll('.model-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
});

// Utility functions
function showToast(message, type = 'info') {
    // Create toast element
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    // Add to toast container
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1055';
        document.body.appendChild(toastContainer);
    }
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Show toast
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard!', 'success');
        }).catch(() => {
            showToast('Failed to copy to clipboard', 'danger');
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showToast('Copied to clipboard!', 'success');
        } catch (err) {
            showToast('Failed to copy to clipboard', 'danger');
        }
        document.body.removeChild(textArea);
    }
}

// Sample data generator for different models
function generateSampleData(modelName) {
    const sampleData = {
        xgb: {
            'feature_Dst_Port': 80,
            'feature_Protocol': 6,
            'feature_Flow_Duration': 1000,
            'feature_Tot_Fwd_Pkts': 10,
            'feature_Tot_Bwd_Pkts': 5,
            'feature_TotLen_Fwd_Pkts': 1000,
            'feature_TotLen_Bwd_Pkts': 500,
            'feature_Fwd_Pkt_Len_Max': 100,
            'feature_Fwd_Pkt_Len_Min': 50,
            'feature_Fwd_Pkt_Len_Mean': 75,
            'feature_Fwd_Pkt_Len_Std': 10,
            'feature_Bwd_Pkt_Len_Max': 100,
            'feature_Bwd_Pkt_Len_Min': 50,
            'feature_Bwd_Pkt_Len_Mean': 75,
            'feature_Bwd_Pkt_Len_Std': 10,
            'feature_Flow_Byts_s': 1000,
            'feature_Flow_Pkts_s': 10,
            'feature_Flow_IAT_Mean': 100,
            'feature_Flow_IAT_Std': 20,
            'feature_Flow_IAT_Max': 200,
            'feature_Flow_IAT_Min': 50,
            'feature_Fwd_IAT_Tot': 1000,
            'feature_Fwd_IAT_Mean': 100,
            'feature_Fwd_IAT_Std': 20,
            'feature_Fwd_IAT_Max': 200,
            'feature_Fwd_IAT_Min': 50,
            'feature_Bwd_IAT_Tot': 500,
            'feature_Bwd_IAT_Mean': 100,
            'feature_Bwd_IAT_Std': 20,
            'feature_Bwd_IAT_Max': 200,
            'feature_Bwd_IAT_Min': 50,
            'feature_Fwd_PSH_Flags': 0,
            'feature_Bwd_PSH_Flags': 0,
            'feature_Fwd_URG_Flags': 0,
            'feature_Bwd_URG_Flags': 0,
            'feature_Fwd_Header_Len': 20,
            'feature_Bwd_Header_Len': 20,
            'feature_Fwd_Pkts_s': 10,
            'feature_Bwd_Pkts_s': 5,
            'feature_Pkt_Len_Min': 50,
            'feature_Pkt_Len_Max': 100,
            'feature_Pkt_Len_Mean': 75,
            'feature_Pkt_Len_Std': 10,
            'feature_Pkt_Len_Var': 100,
            'feature_Fwd_Act_Data_Pkts': 10,
            'feature_Bwd_Act_Data_Pkts': 5,
            'feature_Fwd_Seg_Size_Min': 0,
            'feature_Active_Mean': 100,
            'feature_Active_Std': 20,
            'feature_Active_Max': 200,
            'feature_Active_Min': 50,
            'feature_Idle_Mean': 100,
            'feature_Idle_Std': 20,
            'feature_Idle_Max': 200,
            'feature_Idle_Min': 50
        }
    };
    
    return sampleData[modelName] || {};
}

// Enhanced sample data filling
function fillSampleData() {
    // Try to get features from hidden input first
    const featuresInput = document.getElementById('features-data');
    let features = [];
    
    if (featuresInput && featuresInput.value) {
        features = featuresInput.value.split(',');
    }
    
    if (features.length > 0) {
        // Use template features
        const sampleData = {};
        features.forEach(feature => {
            sampleData['feature_' + feature] = Math.floor(Math.random() * 100) + 1;
        });
        
        Object.keys(sampleData).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = sampleData[key];
            }
        });
    } else {
        // Fallback to hardcoded sample data
        const modelName = document.querySelector('input[name="model_name"]')?.value || 'xgb';
        const sampleData = generateSampleData(modelName);
        
        Object.keys(sampleData).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = sampleData[key];
            }
        });
    }
    
    showToast('Sample data filled!', 'success');
}

// Load CSV and populate form (client-side)
async function loadCsvToForm(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) {
        showToast('No CSV selected', 'warning');
        return;
    }

    try {
        const text = await file.text();
        // Simple CSV parser for first row
        const lines = text.split(/\r?\n/).filter(Boolean);
        if (lines.length === 0) {
            showToast('CSV is empty', 'danger');
            return;
        }

        const header = lines[0].split(',').map(h => h.trim());
        const row = (lines[1] || '').split(',');
        if (row.length === 0 || row.every(v => v === '')) {
            showToast('CSV has no data row', 'danger');
            return;
        }

        // Validate headers against expected features
        const expected = (document.getElementById('features-data')?.value || '').split(',').map(s => s.trim()).filter(Boolean);
        const headerSet = new Set(header);
        const expectedSet = new Set(expected);

        const missing = expected.filter(f => !headerSet.has(f));
        const extra = header.filter(h => !expectedSet.has(h));

        const msgEl = document.getElementById('csvMismatchMsg');
        if (msgEl) {
            msgEl.classList.remove('d-none', 'alert-success', 'alert-warning');
            msgEl.classList.add('alert', 'alert-permanent');
            if (missing.length === 0 && extra.length === 0) {
                msgEl.classList.remove('alert-warning');
                msgEl.classList.add('alert-success');
                msgEl.textContent = 'Headers match schema.';
            } else {
                msgEl.classList.remove('alert-success');
                msgEl.classList.add('alert-warning');
                const parts = [];
                if (missing.length) parts.push(`Missing (${missing.length}): ${missing.slice(0,10).join(', ')}${missing.length>10?'...':''}`);
                if (extra.length) parts.push(`Extra (${extra.length}): ${extra.slice(0,10).join(', ')}${extra.length>10?'...':''}`);
                msgEl.textContent = parts.join(' | ');
            }
        }

        // Map header->value and fill matching feature inputs only
        const valueByCol = {};
        header.forEach((h, i) => {
            valueByCol[h] = row[i];
        });

        let filled = 0;
        expected.forEach(col => {
            const input = document.getElementById('feature_' + col);
            if (input) {
                const val = valueByCol[col];
                if (val !== undefined && val !== '') {
                    const num = Number(val);
                    input.value = Number.isFinite(num) ? num : '';
                    filled += 1;
                }
            }
        });

        if (filled === 0) {
            showToast('CSV headers do not match expected features', 'warning');
        } else {
            showToast(`Loaded CSV: populated ${filled} fields`, 'success');
        }
    } catch (err) {
        showToast('Failed to read CSV', 'danger');
    }
}

// Clear all feature inputs
function clearFeatureForm() {
    const inputs = document.querySelectorAll('[id^="feature_"]');
    inputs.forEach(inp => {
        inp.value = '';
    });
    showToast('Cleared inputs', 'info');
}

