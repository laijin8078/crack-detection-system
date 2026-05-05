// 建筑裂缝检测系统 - 前端JavaScript

let selectedFile = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initUpload();
    initConfidenceSlider();
    loadHistory();
    loadStatistics();

    // 定时刷新历史记录和统计信息
    setInterval(loadHistory, 30000);  // 30秒
    setInterval(loadStatistics, 60000);  // 60秒
});

// 初始化上传功能
function initUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const detectBtn = document.getElementById('detectBtn');

    // 点击上传区域
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });

    // 检测按钮
    detectBtn.addEventListener('click', detectCrack);
}

// 处理文件选择
function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('请选择图像文件');
        return;
    }

    selectedFile = file;

    // 预览图像
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewArea = document.getElementById('previewArea');
        previewArea.innerHTML = `<img src="${e.target.result}" alt="预览">`;
        document.getElementById('detectBtn').disabled = false;
    };
    reader.readAsDataURL(file);
}

// 初始化置信度滑块
function initConfidenceSlider() {
    const slider = document.getElementById('confThreshold');
    const valueDisplay = document.getElementById('confValue');

    slider.addEventListener('input', (e) => {
        valueDisplay.textContent = e.target.value;
    });
}

// 检测裂缝
async function detectCrack() {
    if (!selectedFile) {
        alert('请先选择图像');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    const originalText = detectBtn.innerHTML;
    detectBtn.disabled = true;
    detectBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 检测中...';

    const formData = new FormData();
    formData.append('file', selectedFile);

    const confThreshold = document.getElementById('confThreshold').value;

    try {
        const response = await fetch(`/api/detect?conf_threshold=${confThreshold}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            displayResult(result);
            loadHistory();  // 刷新历史记录
            loadStatistics();  // 刷新统计信息
        } else {
            alert('检测失败: ' + (result.detail || '未知错误'));
        }
    } catch (error) {
        alert('检测失败: ' + error.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.innerHTML = originalText;
    }
}

// 显示检测结果
function displayResult(result) {
    const resultArea = document.getElementById('resultArea');
    const resultImage = document.getElementById('resultImage');
    const resultDetails = document.getElementById('resultDetails');

    // 显示结果图像
    resultImage.innerHTML = `<img src="${result.result_image_url}" alt="检测结果">`;

    // 显示检测详情
    let detailsHTML = `
        <div class="alert alert-info">
            <h6><i class="fas fa-info-circle"></i> 检测信息</h6>
            <p><strong>检测ID:</strong> ${result.detection_id}</p>
            <p><strong>处理时间:</strong> ${result.processing_time}秒</p>
            <p><strong>裂缝数量:</strong> ${result.num_cracks}</p>
        </div>
    `;

    if (result.detections && result.detections.length > 0) {
        detailsHTML += '<h6>裂缝详情:</h6>';
        result.detections.forEach((det, index) => {
            detailsHTML += `
                <div class="detection-item">
                    <h6>裂缝 ${index + 1}</h6>
                    <p><strong>类别:</strong> ${det.class}</p>
                    <p><strong>置信度:</strong> ${(det.confidence * 100).toFixed(2)}%</p>
                    <p><strong>位置:</strong> (${det.center.x.toFixed(0)}, ${det.center.y.toFixed(0)})</p>
                    <p><strong>尺寸:</strong> ${det.size.width.toFixed(0)} × ${det.size.height.toFixed(0)}</p>
                </div>
            `;
        });
    } else {
        detailsHTML += '<p class="text-muted">未检测到裂缝</p>';
    }

    resultDetails.innerHTML = detailsHTML;
    resultArea.style.display = 'block';

    // 滚动到结果区域
    resultArea.scrollIntoView({ behavior: 'smooth' });
}

// 加载历史记录
async function loadHistory() {
    try {
        const response = await fetch('/api/detections?limit=10');
        const result = await response.json();

        if (result.success) {
            displayHistory(result.detections);
        }
    } catch (error) {
        console.error('加载历史记录失败:', error);
    }
}

// 显示历史记录
function displayHistory(detections) {
    const historyTable = document.getElementById('historyTable');

    if (!detections || detections.length === 0) {
        historyTable.innerHTML = '<tr><td colspan="7" class="text-center">暂无记录</td></tr>';
        return;
    }

    let html = '';
    detections.forEach(det => {
        const timestamp = new Date(det.timestamp).toLocaleString('zh-CN');
        html += `
            <tr>
                <td>${det.id}</td>
                <td>${timestamp}</td>
                <td>${det.image_name}</td>
                <td><span class="badge bg-primary">${det.num_cracks}</span></td>
                <td>${(det.avg_confidence * 100).toFixed(2)}%</td>
                <td>${det.processing_time ? det.processing_time.toFixed(3) + 's' : '-'}</td>
                <td>
                    <button class="btn btn-sm btn-info" onclick="viewDetail(${det.id})">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    });

    historyTable.innerHTML = html;
}

// 查看详情
async function viewDetail(detectionId) {
    try {
        const response = await fetch(`/api/detection/${detectionId}`);
        const result = await response.json();

        if (result.success) {
            // 这里可以显示详细信息的模态框
            alert('详情功能待实现');
        }
    } catch (error) {
        alert('加载详情失败: ' + error.message);
    }
}

// 加载统计信息
async function loadStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const result = await response.json();

        if (result.success) {
            displayStatistics(result.statistics);
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

// 显示统计信息
function displayStatistics(stats) {
    const overall = stats.overall;

    document.getElementById('totalImages').textContent = overall.total_images || 0;
    document.getElementById('totalCracks').textContent = overall.total_cracks || 0;
    document.getElementById('avgCracks').textContent = overall.avg_cracks_per_image ?
        overall.avg_cracks_per_image.toFixed(2) : '0.00';
    document.getElementById('avgConfidence').textContent = overall.avg_confidence ?
        (overall.avg_confidence * 100).toFixed(2) + '%' : '0%';
}
