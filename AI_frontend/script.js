// ==================== API settings ====================
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000', 
    PREDICT_ENDPOINT: '/predict',
    ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/jpg']
};

// ==================== variables declaration ====================
let dropZone, fileInput, uploadBtn, previewSection, imagePreview;
let fileNameElement, progressBar, analyzeBtn, clearBtn;
let resultsPlaceholder, resultsContent;

// ==================== initialize everything ====================
document.addEventListener('DOMContentLoaded', function() {
    // set the date
    const now = new Date();
    document.getElementById('current-date').textContent = now.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    // extract DOM elements
    dropZone = document.getElementById('drop-area');
    fileInput = document.getElementById('file-input');
    uploadBtn = document.getElementById('upload-btn');
    previewSection = document.getElementById('preview-section');
    imagePreview = document.getElementById('image-preview');
    fileNameElement = document.getElementById('file-name');
    progressBar = document.getElementById('progress-bar');
    analyzeBtn = document.getElementById('analyze-btn');
    clearBtn = document.getElementById('clear-btn');
    resultsPlaceholder = document.getElementById('results-placeholder');
    resultsContent = document.getElementById('results-content');

    // PDF download button listener
    const downloadBtn = document.getElementById('download-report');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            if (resultsContent.classList.contains('d-none')) {
                alert('Please analyze an image first before downloading the report.');
                return;
            }
            generatePDFReport();
        });
    }

    // upload button
    uploadBtn.addEventListener('click', () => fileInput.click());

    // file input button
    fileInput.addEventListener('change', handleFileSelect);

    // drag funciton
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    // clear button
    clearBtn.addEventListener('click', function() {
        resetAnalysisState();
        fileInput.value = '';
        previewSection.classList.add('d-none');
        fileNameElement.textContent = '';
    });

    // ==================== analyze button functions ====================
    analyzeBtn.addEventListener('click', async function() {
        // extract files
        if (!fileInput.files[0]) {
            alert('Please select an image first!');
            return;
        }
        
        const file = fileInput.files[0];
        
        // check file type
        if (!API_CONFIG.ALLOWED_TYPES.includes(file.type)) {
            alert('Only JPEG or PNG images are supported!');
            return;
        }
        
        // show loading animation
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>AI Analyzing...';
        analyzeBtn.disabled = true;
        
        // show progress bar
        progressBar.classList.remove('d-none');
        const progressBarElement = progressBar.querySelector('.progress-bar');
        progressBarElement.style.width = '30%';
        progressBarElement.textContent = '30% - Uploading...';
        
        try {
            // prepare FormData
            const formData = new FormData();
            formData.append('file', file);
            
            // send request to backend API
            const API_URL = API_CONFIG.BASE_URL + API_CONFIG.PREDICT_ENDPOINT;
            console.log('Sending request to:', API_URL);
            
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });
            
            // updating progress
            progressBarElement.style.width = '70%';
            progressBarElement.textContent = '70% - Processing...';
            
            // check the response
            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    errorMessage = await response.text();
                }
                throw new Error(errorMessage);
            }
            
            // check JSON response
            const result = await response.json();
            console.log('API Response:', result);
            
            // update progress status (finished)
            progressBarElement.style.width = '100%';
            progressBarElement.textContent = '100% - Complete!';
            
            // show results
            setTimeout(() => {
                showYOLOResults(result);
                analyzeBtn.innerHTML = '<i class="fas fa-check me-2"></i>Analysis Complete';
                analyzeBtn.disabled = false;
                analyzeBtn.classList.remove('btn-success');
                analyzeBtn.classList.add('btn-secondary');
                progressBar.classList.add('d-none');
            }, 500);
            
        } catch (error) {
            // prompt errors
            console.error('API Error:', error);
            
            // restore progress bar
            const progressBarElement = progressBar.querySelector('.progress-bar');
            progressBarElement.style.width = '0%';
            progressBarElement.textContent = 'Failed';
            
            // show error msg
            let errorMsg = error.message;
            if (error.message.includes('Failed to fetch')) {
                errorMsg = 'Cannot connect to server. Make sure: 1) Backend is running, 2) Correct URL, 3) No CORS error';
            }
            
            // window to show error
            const errorAlert = document.createElement('div');
            errorAlert.className = 'alert alert-danger alert-dismissible fade show mt-3';
            errorAlert.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>API Error:</strong> ${errorMsg}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.querySelector('.container').insertBefore(errorAlert, document.querySelector('.container').firstChild);
            
            // Reset analyze button
            analyzeBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Try Again';
            analyzeBtn.disabled = false;
            analyzeBtn.classList.add('btn-success');
            progressBar.classList.add('d-none');
        }
    });
});

// ==================== handle drop files function ====================
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        
        // AUTO-CLEAR: Reset everything when new file is uploaded
        resetAnalysisState();
        
        // show file name
        fileNameElement.textContent = `Selected: ${file.name}`;
        
        // show progress bar
        progressBar.classList.remove('d-none');
        simulateUpload();
        
        // clear old boxes
        const oldBoxes = document.querySelectorAll('.bounding-box');
        oldBoxes.forEach(box => box.remove());
        
        // preview image if available
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewSection.classList.remove('d-none');
                
                // Reset analyze button for new file
                analyzeBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Start AI Analysis';
                analyzeBtn.disabled = false;
                analyzeBtn.classList.remove('btn-secondary');
                analyzeBtn.classList.add('btn-success');
            };
            reader.readAsDataURL(file);
        } else {
            // show preview DICOM files if can
            imagePreview.src = 'https://via.placeholder.com/400x300/2c80ff/ffffff?text=DICOM+MRI+Scan';
            previewSection.classList.remove('d-none');
            
            // Reset analyze button for new file
            analyzeBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Start AI Analysis';
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('btn-secondary');
            analyzeBtn.classList.add('btn-success');
        }
    }
}

// New function to reset analysis state
function resetAnalysisState() {
    // Hide results and show placeholder
    resultsContent.classList.add('d-none');
    resultsPlaceholder.classList.remove('d-none');
    
    // Reset analyze button
    analyzeBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Start AI Analysis';
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('btn-secondary');
    analyzeBtn.classList.add('btn-success');
    
    // Hide progress bar
    progressBar.classList.add('d-none');
    
    // Clear any existing bounding boxes
    const oldBoxes = document.querySelectorAll('.bounding-box');
    oldBoxes.forEach(box => box.remove());
}

function simulateUpload() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        const progressBarElement = progressBar.querySelector('.progress-bar');
        progressBarElement.style.width = `${progress}%`;
        progressBarElement.textContent = `${progress}%`;
        
        if (progress >= 100) {
            clearInterval(interval);
            progressBarElement.textContent = 'Upload Complete!';
            setTimeout(() => {
                progressBar.classList.add('d-none');
            }, 1000);
        }
    }, 200);
}

// ==================== showing YOLO results function ====================
function showYOLOResults(apiResult) {
    // hide placeholder and show results
    resultsPlaceholder.classList.add('d-none');
    resultsContent.classList.remove('d-none');
    
    // update detected results
    const detections = apiResult.detections || [];
    let primaryDisease = 'No Disease Detected';
    let highestConfidence = 0;
    let diseaseCounts = {};
    
    // Log what we're getting
    debugBoundingBoxes(detections);
    
    // show what diseases that captured
    detections.forEach(det => {
        const diseaseName = formatDiseaseName(det.label);
        const confidence = det.confidence;
        
        diseaseCounts[diseaseName] = (diseaseCounts[diseaseName] || 0) + 1;
        
        if (confidence > highestConfidence) {
            highestConfidence = confidence;
            primaryDisease = diseaseName;
        }
    });
    
    // summarize if have multiple counts
    if (Object.keys(diseaseCounts).length > 0) {
        const diseaseList = Object.entries(diseaseCounts)
            .map(([name, count]) => `${name} (${count})`)
            .join(', ');
        primaryDisease = `${primaryDisease} | Detected: ${diseaseList}`;
    }
    
    // update content
    document.getElementById('disease-name').textContent = primaryDisease;
    document.getElementById('confidence-badge').textContent = `${Math.round(highestConfidence * 100)}%`;
    
    // set severity according to confidence
    let severity, severityColor;
    if (highestConfidence > 0.8) {
        severity = 'High Risk';
        severityColor = 'text-danger';
    } else if (highestConfidence > 0.5) {
        severity = 'Moderate Risk';
        severityColor = 'text-warning';
    } else if (highestConfidence > 0.2) {
        severity = 'Low Risk';
        severityColor = 'text-success';
    } else {
        severity = 'No Significant Risk';
        severityColor = 'text-secondary';
    }
    
    document.getElementById('severity').textContent = severity;
    document.getElementById('severity').className = severityColor;
    document.getElementById('icd-code').textContent = getICDCode(primaryDisease);
    
    // update lesions table
    const lesionsTable = document.getElementById('lesions-table');
    lesionsTable.innerHTML = '';
    
    if (detections.length > 0) {
        detections.forEach((detection, index) => {
            const row = document.createElement('tr');
            const diseaseName = formatDiseaseName(detection.label);
            const box = detection.box;
            const size = calculateLesionSize(box);
            
            row.innerHTML = `
                <td>${diseaseName}</td>
                <td>Lesion ${index + 1}</td>
                <td>${size}</td>
                <td><span class="badge ${detection.confidence > 0.8 ? 'bg-success' : detection.confidence > 0.5 ? 'bg-warning' : 'bg-secondary'}">
                    ${Math.round(detection.confidence * 100)}%
                </span></td>
            `;
            lesionsTable.appendChild(row);
        });
    } else {
        lesionsTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center text-muted">
                    <i class="fas fa-check-circle me-2"></i>No abnormalities detected
                </td>
            </tr>
        `;
    }
    
    // update recommendations list
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = '';
    
    const recommendations = generateRecommendations(detections, highestConfidence);
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });
    
    // just to show bounding boxes on image
    showBoundingBoxesOnImage(detections);
}

// ==================== function for formatting ====================
function formatDiseaseName(label) {
    const diseaseMap = {
        'pneumonia': 'Pneumonia',
        'cancer': 'Lung Cancer',
        'nodule': 'Lung Nodule',
        'tuberculosis': 'Tuberculosis',
        'covid': 'COVID-19',
        'fibrosis': 'Pulmonary Fibrosis',
        'emphysema': 'Emphysema'
    };
    
    return diseaseMap[label.toLowerCase()] || label.charAt(0).toUpperCase() + label.slice(1);
}

function getICDCode(diseaseName) {
    const icdMap = {
        'Pneumonia': 'J18.9',
        'Lung Cancer': 'C34.9',
        'Lung Nodule': 'R91.8',
        'Tuberculosis': 'A15.9',
        'COVID-19': 'U07.1',
        'Pulmonary Fibrosis': 'J84.1',
        'Emphysema': 'J43.9'
    };
    
    for (const [key, code] of Object.entries(icdMap)) {
        if (diseaseName.includes(key)) return code;
    }
    return 'R09.8';
}

function calculateLesionSize(box) {
    if (!box || box.length !== 4) return 'N/A';
    const [x1, y1, x2, y2] = box;
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);
    const area = width * height;
    
    if (area < 1000) return 'Small (< 1cm)';
    if (area < 5000) return 'Medium (1-2cm)';
    return 'Large (> 2cm)';
}

function generateRecommendations(detections, confidence) {
    const recommendations = [];
    
    if (detections.length === 0) {
        recommendations.push(
            'No abnormalities detected in this scan',
            'Routine follow-up recommended in 1 year',
            'Maintain healthy lifestyle'
        );
        return recommendations;
    }
    
    if (detections.some(d => formatDiseaseName(d.label).includes('Cancer'))) {
        recommendations.push(
            'Immediate consultation with oncologist required',
            'Biopsy recommended for confirmation',
            'PET-CT scan advised for staging'
        );
    }
    
    if (detections.some(d => formatDiseaseName(d.label).includes('Pneumonia'))) {
        recommendations.push(
            'Antibiotic treatment recommended',
            'Chest X-ray follow-up in 2 weeks',
            'Monitor fever and respiratory symptoms'
        );
    }
    
    if (detections.some(d => formatDiseaseName(d.label).includes('Nodule'))) {
        recommendations.push(
            'Follow-up CT scan in 3-6 months',
            'Monitor for size changes',
            'Consider biopsy if growing'
        );
    }
    
    if (recommendations.length === 0) {
        recommendations.push(
            'Consult with pulmonology specialist',
            'Further diagnostic tests may be needed',
            'Regular monitoring recommended'
        );
    }
    
    recommendations.push('This AI analysis should be reviewed by a medical professional');
    
    return recommendations;
}

function showBoundingBoxesOnImage(detections) {
    const img = document.getElementById('image-preview');
    if (!img || detections.length === 0) return;
    
    const oldBoxes = document.querySelectorAll('.bounding-box');
    oldBoxes.forEach(box => box.remove());
    
    // Get the displayed image dimensions (not natural dimensions)
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    
    // Get natural dimensions for scaling
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    console.log('Image dimensions:', { 
        display: { displayWidth, displayHeight }, 
        natural: { naturalWidth, naturalHeight } 
    });
    console.log('Detections:', detections);
    
    detections.forEach((det, index) => {
        const box = det.box;
        if (!box || box.length !== 4) {
            console.warn('Invalid box format:', box);
            return;
        }
        
        let [x1, y1, x2, y2] = box;
        console.log(`Original box ${index}:`, { x1, y1, x2, y2 });
        
        // STRATEGY 1: Check if coordinates might be for a different scale
        // If coordinates are huge (like > 1000), they might be for full resolution
        if (x2 > 1000 || y2 > 1000) {
            console.log('Large coordinates detected, assuming full resolution');
            if (naturalWidth > 0 && naturalHeight > 0) {
                // Scale down to display size
                const scaleX = displayWidth / naturalWidth;
                const scaleY = displayHeight / naturalHeight;
                x1 = x1 * scaleX;
                y1 = y1 * scaleY;
                x2 = x2 * scaleX;
                y2 = y2 * scaleY;
                console.log(`Scaled to display:`, { x1, y1, x2, y2 });
            }
        }
        // STRATEGY 2: Check if normalized (0-1)
        else if (x2 <= 1 && y2 <= 1) {
            console.log('Normalized coordinates detected');
            x1 = x1 * displayWidth;
            y1 = y1 * displayHeight;
            x2 = x2 * displayWidth;
            y2 = y2 * displayHeight;
        }
        // STRATEGY 3: Already in display coordinates (do nothing)
        
        // Apply bounds checking
        x1 = Math.max(0, Math.min(x1, displayWidth - 1));
        y1 = Math.max(0, Math.min(y1, displayHeight - 1));
        x2 = Math.max(0, Math.min(x2, displayWidth));
        y2 = Math.max(0, Math.min(y2, displayHeight));
        
        // Ensure width/height are positive
        const width = Math.max(1, x2 - x1);
        const height = Math.max(1, y2 - y1);
        
        console.log(`Final box ${index}:`, { x1, y1, width, height });
        
        // Create bounding box
        const boxDiv = document.createElement('div');
        boxDiv.className = 'bounding-box';
        boxDiv.style.cssText = `
            position: absolute;
            left: ${x1}px;
            top: ${y1}px;
            width: ${width}px;
            height: ${height}px;
            border: 3px solid ${det.confidence > 0.8 ? '#ff0000' : det.confidence > 0.5 ? '#ff9900' : '#00ff00'};
            background-color: ${det.confidence > 0.8 ? 'rgba(255, 0, 0, 0.1)' : 
                              det.confidence > 0.5 ? 'rgba(255, 153, 0, 0.1)' : 
                              'rgba(0, 255, 0, 0.1)'};
            pointer-events: none;
            z-index: 10;
        `;
        
        // Create label
        const label = document.createElement('div');
        label.textContent = `${formatDiseaseName(det.label)} ${Math.round(det.confidence * 100)}%`;
        label.style.cssText = `
            position: absolute;
            top: -28px;
            left: 0;
            background: ${det.confidence > 0.8 ? '#ff0000' : det.confidence > 0.5 ? '#ff9900' : '#00ff00'};
            color: white;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: bold;
            border-radius: 4px;
            white-space: nowrap;
            z-index: 20;
        `;
        boxDiv.appendChild(label);
        
        // Add to container
        const container = img.parentElement;
        if (!container.style.position || container.style.position === 'static') {
            container.style.position = 'relative';
        }
        container.appendChild(boxDiv);
    });
}

function debugBoundingBoxes(detections) {
    console.log('=== BOUNDING BOX DEBUG ===');
    detections.forEach((det, index) => {
        console.log(`Detection ${index}:`, {
            label: det.label,
            confidence: det.confidence,
            box: det.box,
            boxType: Array.isArray(det.box) ? 'array' : typeof det.box,
            boxLength: Array.isArray(det.box) ? det.box.length : 'N/A'
        });
    });
}

// ==================== PDF Report Generation ====================
function generatePDFReport() {
    console.log('Generating PDF report...');
    
    // Check if jsPDF is loaded
    if (typeof window.jspdf === 'undefined') {
        alert('PDF library not loaded. Please refresh the page and make sure jsPDF is included.');
        console.error('jsPDF is undefined. Check if the script is loaded.');
        return;
    }
    
    try {
        // Use the global jsPDF
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Get current data
        const diseaseName = document.getElementById('disease-name').textContent;
        const confidence = document.getElementById('confidence-badge').textContent;
        const severity = document.getElementById('severity').textContent;
        const icdCode = document.getElementById('icd-code').textContent;
        
        console.log('Data for PDF:', { diseaseName, confidence, severity, icdCode });
        
        // Title
        doc.setFontSize(20);
        doc.setTextColor(44, 128, 255);
        doc.text('LungScan AI - Medical Report', 20, 20);
        
        // Report info
        doc.setFontSize(11);
        doc.setTextColor(0, 0, 0);
        doc.text(`Report Date: ${new Date().toLocaleDateString()}`, 20, 35);
        doc.text(`Patient ID: DEMO-${Date.now().toString().slice(-6)}`, 20, 42);
        
        // Diagnosis section
        doc.setFontSize(16);
        doc.setTextColor(0, 0, 0);
        doc.text('Primary Diagnosis', 20, 60);
        
        doc.setFontSize(12);
        doc.text(`Disease: ${diseaseName}`, 20, 70);
        doc.text(`Confidence: ${confidence}`, 20, 77);
        doc.text(`Severity: ${severity}`, 20, 84);
        doc.text(`ICD-10 Code: ${icdCode}`, 20, 91);
        
        // Lesions table
        doc.setFontSize(16);
        doc.text('Detected Lesions', 20, 110);
        
        const lesionsTable = document.getElementById('lesions-table');
        const rows = lesionsTable.querySelectorAll('tr');
        let yPos = 120;
        
        // Table headers
        doc.setFontSize(10);
        doc.text('Type', 20, yPos);
        doc.text('Location', 60, yPos);
        doc.text('Size', 110, yPos);
        doc.text('Confidence', 150, yPos);
        yPos += 7;
        
        // Draw line
        doc.line(20, yPos, 180, yPos);
        yPos += 5;
        
        // Table rows
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length === 4) {
                doc.text(cells[0].textContent || '', 20, yPos);
                doc.text(cells[1].textContent || '', 60, yPos);
                doc.text(cells[2].textContent || '', 110, yPos);
                doc.text(cells[3].textContent || '', 150, yPos);
                yPos += 7;
            }
        });
        
        // Recommendations
        doc.setFontSize(16);
        doc.text('Clinical Recommendations', 20, yPos + 10);
        yPos += 20;
        
        doc.setFontSize(10);
        const recommendationsList = document.getElementById('recommendations-list');
        const recommendations = recommendationsList.querySelectorAll('li');
        
        recommendations.forEach((rec, index) => {
            if (yPos > 270) {
                doc.addPage();
                yPos = 20;
            }
            const text = rec.textContent || '';
            doc.text(`• ${text}`, 20, yPos);
            yPos += 7;
        });
        
        // Footer
        doc.setFontSize(8);
        doc.setTextColor(100, 100, 100);
        doc.text('This report was generated by LungScan AI system.', 20, 280);
        doc.text('For academic demonstration purposes only. Not for clinical use.', 20, 285);
        
        // Save the PDF
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `LungScan_Report_${timestamp}.pdf`;
        console.log('Saving PDF as:', filename);
        doc.save(filename);
        
        console.log('PDF generated successfully!');
        
    } catch (error) {
        console.error('PDF generation error:', error);
        alert('Error generating PDF: ' + error.message);
    }
}
