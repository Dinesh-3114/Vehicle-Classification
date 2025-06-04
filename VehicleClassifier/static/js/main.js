// Global variables
let selectedFiles = [];
let uploadInProgress = false;

// Initialize drag and drop functionality
function initializeDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    if (!dropZone || !fileInput) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    
    // Handle click to open file dialog
    dropZone.addEventListener('click', () => {
        if (!uploadInProgress) {
            fileInput.click();
        }
    });
}

// Initialize file input functionality
function initializeFileInput() {
    const fileInput = document.getElementById('fileInput');
    const clearButton = document.getElementById('clearFiles');
    const uploadForm = document.getElementById('uploadForm');
    
    if (!fileInput) return;

    fileInput.addEventListener('change', function(e) {
        handleFileSelection(e.target.files);
    });
    
    if (clearButton) {
        clearButton.addEventListener('click', clearFileSelection);
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }
}

// Prevent default drag behaviors
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop zone
function highlight(e) {
    const dropZone = document.getElementById('dropZone');
    if (dropZone && !uploadInProgress) {
        dropZone.classList.add('dragover');
    }
}

// Remove highlight from drop zone
function unhighlight(e) {
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        dropZone.classList.remove('dragover');
    }
}

// Handle dropped files
function handleDrop(e) {
    if (uploadInProgress) return;
    
    const dt = e.dataTransfer;
    const files = dt.files;
    
    handleFileSelection(files);
}

// Handle file selection (both drag & drop and file input)
function handleFileSelection(files) {
    if (uploadInProgress) return;
    
    const validFiles = Array.from(files).filter(file => {
        // Check file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showToast(`File "${file.name}" is not a valid image format.`, 'warning');
            return false;
        }
        
        // Check file size (16MB limit)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showToast(`File "${file.name}" is too large. Maximum size is 16MB.`, 'warning');
            return false;
        }
        
        return true;
    });
    
    if (validFiles.length === 0) {
        return;
    }
    
    // Add new files to selection (avoid duplicates)
    validFiles.forEach(file => {
        if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
        }
    });
    
    updateFilePreview();
    updateUploadButton();
}

// Update file preview display
function updateFilePreview() {
    const filePreview = document.getElementById('filePreview');
    const fileList = document.getElementById('fileList');
    const clearButton = document.getElementById('clearFiles');
    
    if (!filePreview || !fileList) return;
    
    if (selectedFiles.length === 0) {
        filePreview.style.display = 'none';
        if (clearButton) clearButton.style.display = 'none';
        return;
    }
    
    filePreview.style.display = 'block';
    if (clearButton) clearButton.style.display = 'inline-block';
    
    fileList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <i data-feather="image"></i>
                <span class="file-name">${file.name}</span>
            </div>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button type="button" class="remove-file" onclick="removeFile(${index})" title="Remove file">
                <i data-feather="x"></i>
            </button>
        `;
        fileList.appendChild(fileItem);
    });
    
    // Re-initialize feather icons
    feather.replace();
}

// Remove file from selection
function removeFile(index) {
    if (uploadInProgress) return;
    
    selectedFiles.splice(index, 1);
    updateFilePreview();
    updateUploadButton();
}

// Clear all selected files
function clearFileSelection() {
    if (uploadInProgress) return;
    
    selectedFiles = [];
    document.getElementById('fileInput').value = '';
    updateFilePreview();
    updateUploadButton();
}

// Update upload button state
function updateUploadButton() {
    const uploadBtn = document.getElementById('uploadBtn');
    if (!uploadBtn) return;
    
    uploadBtn.disabled = selectedFiles.length === 0 || uploadInProgress;
    
    if (selectedFiles.length > 0 && !uploadInProgress) {
        uploadBtn.innerHTML = `
            <i data-feather="send" class="me-1"></i>
            Classify ${selectedFiles.length} Image${selectedFiles.length > 1 ? 's' : ''}
        `;
    } else {
        uploadBtn.innerHTML = `
            <i data-feather="send" class="me-1"></i>
            Classify Images
        `;
    }
    
    feather.replace();
}

// Handle form submission
function handleFormSubmit(e) {
    if (selectedFiles.length === 0) {
        e.preventDefault();
        showToast('Please select at least one image file.', 'warning');
        return;
    }
    
    if (uploadInProgress) {
        e.preventDefault();
        return;
    }
    
    // Update file input with selected files
    const fileInput = document.getElementById('fileInput');
    const dataTransfer = new DataTransfer();
    
    selectedFiles.forEach(file => {
        dataTransfer.items.add(file);
    });
    
    fileInput.files = dataTransfer.files;
    
    // Show progress
    showUploadProgress();
    uploadInProgress = true;
    updateUploadButton();
}

// Show upload progress
function showUploadProgress() {
    const progressContainer = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (!progressContainer) return;
    
    progressContainer.style.display = 'block';
    
    if (uploadBtn) {
        uploadBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Processing...
        `;
        uploadBtn.disabled = true;
    }
    
    // Simulate progress (in real implementation, this would be updated by server)
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        
        if (progressBar) progressBar.style.width = progress + '%';
        if (progressText) progressText.textContent = Math.round(progress) + '%';
    }, 500);
    
    // Store interval for cleanup
    window.uploadProgressInterval = interval;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    toast.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
    `;
    
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, 5000);
}

// Utility function to get confidence color class
function getConfidenceColorClass(confidence) {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'danger';
}

// Initialize page-specific functionality
function initializePage() {
    const currentPage = window.location.pathname;
    
    if (currentPage === '/' || currentPage.includes('index')) {
        initializeDragAndDrop();
        initializeFileInput();
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    
    // Initialize all feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
    
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Clean up intervals on page unload
window.addEventListener('beforeunload', function() {
    if (window.uploadProgressInterval) {
        clearInterval(window.uploadProgressInterval);
    }
});

// Handle page visibility change (for progress updates)
document.addEventListener('visibilitychange', function() {
    if (document.hidden && window.uploadProgressInterval) {
        clearInterval(window.uploadProgressInterval);
    }
});
