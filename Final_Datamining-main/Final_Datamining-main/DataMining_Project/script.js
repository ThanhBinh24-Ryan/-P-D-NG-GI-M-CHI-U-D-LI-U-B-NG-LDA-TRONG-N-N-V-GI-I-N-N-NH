document.addEventListener('DOMContentLoaded', () => {
    // Các thành phần DOM
    const imageInput = document.getElementById('imageInput');
    const originalImage = document.getElementById('originalImage');
    const compressedImage = document.getElementById('compressedImage');
    const decompressedImage = document.getElementById('decompressedImage');
    const noImageMsg = document.getElementById('noImageMsg');
    const originalDimensions = document.getElementById('originalDimensions');
    
    // Các nút điều khiển
    const functionButtons = document.getElementById('functionButtons');
    const selectLdaBtn = document.getElementById('selectLdaBtn');
    const selectPcaBtn = document.getElementById('selectPcaBtn');
    const compressBtn = document.getElementById('compressBtn');
    const decompressBtn = document.getElementById('decompressBtn');
    const toggleStatsBtn = document.getElementById('toggleStatsBtn');
    const saveCompressedBtn = document.getElementById('saveCompressedBtn');
    const saveDecompressedBtn = document.getElementById('saveDecompressedBtn');
    
    // Loaders và stats
    const compressLoader = document.getElementById('compressLoader');
    const decompressLoader = document.getElementById('decompressLoader');
    const statsPanel = document.getElementById('statsPanel');
    
    // Các trường thống kê
    const originalSize = document.getElementById('originalSize');
    const compressedSize = document.getElementById('compressedSize');
    const compressionRatio = document.getElementById('compressionRatio');
    const compressionTime = document.getElementById('compressionTime');
    const decompressionTime = document.getElementById('decompressionTime');
    const algorithm = document.getElementById('algorithm');
    const componentsCount = document.getElementById('componentsCount');
    const varianceExplained = document.getElementById('varianceExplained');

    // Các biến trạng thái
    let selectedImageData = null;
    let compressedImageData = null;
    let decompressedImageData = null;
    let currentAlgorithm = null;
    let currentStats = null;
    let imageOriginalDimensions = null;

    // Hàm tiện ích để format kích thước file
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Xử lý khi người dùng chọn ảnh
    imageInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                selectedImageData = e.target.result;
                originalImage.src = selectedImageData;
                noImageMsg.style.display = 'none';
                originalImage.style.display = 'block';
                
                // Lấy kích thước ảnh và hiển thị
                const img = new Image();
                img.onload = () => {
                    imageOriginalDimensions = {
                        width: img.width,
                        height: img.height
                    };
                    originalDimensions.textContent = `Kích thước: ${img.width} x ${img.height} px`;
                };
                img.src = selectedImageData;
                
                // Hiển thị các nút chức năng
                functionButtons.classList.remove('hidden');
                
                // Reset các trạng thái khác
                resetProcessingState();
            };
            reader.readAsDataURL(file);
        }
    });

    // Reset trạng thái khi chọn ảnh mới
    function resetProcessingState() {
        compressedImage.src = '';
        decompressedImage.src = '';
        compressBtn.disabled = true;
        decompressBtn.disabled = true;
        saveCompressedBtn.disabled = true;
        saveDecompressedBtn.disabled = true;
        toggleStatsBtn.disabled = true;
        statsPanel.classList.add('hidden');
        currentAlgorithm = null;
        currentStats = null;
        
        // Reset style cho các nút
        selectLdaBtn.classList.remove('shadow-lg', 'bg-green-700', 'ring-4', 'ring-green-700', 'font-bold');
        selectPcaBtn.classList.remove('shadow-lg', 'bg-purple-700', 'ring-4', 'ring-purple-700', 'font-bold');
        selectLdaBtn.classList.add('bg-green-500', 'hover:bg-green-600');
        selectPcaBtn.classList.add('bg-purple-500', 'hover:bg-purple-600');
        
        // Reset thống kê
        resetStats();
    }

    // Reset các giá trị thống kê
    function resetStats() {
        originalSize.textContent = '-';
        compressedSize.textContent = '-';
        compressionRatio.textContent = '-';
        compressionTime.textContent = '-';
        decompressionTime.textContent = '-';
        algorithm.textContent = '-';
        componentsCount.textContent = '-';
        varianceExplained.textContent = '-';
    }

    // Hàm cập nhật thống kê từ dữ liệu từ API
    function updateStats(stats) {
        if (!stats) return;
        
        currentStats = stats;
        
        // Cập nhật UI với thông tin thống kê
        originalSize.textContent = formatFileSize(stats.original_size || 0);
        compressedSize.textContent = formatFileSize(stats.compressed_size || 0);
        compressionRatio.textContent = stats.compression_ratio ? `${stats.compression_ratio}:1` : '-';
        compressionTime.textContent = stats.processing_time ? `${stats.processing_time} giây` : '-';
        decompressionTime.textContent = stats.decompression_time ? `${stats.decompression_time} giây` : '-';
        algorithm.textContent = stats.algorithm ? stats.algorithm.toUpperCase() : '-';
        componentsCount.textContent = stats.components || '-';
        
        // Hiển thị phương sai giải thích (chỉ có ở PCA)
        if (stats.algorithm === 'PCA' && stats.variance_explained) {
            varianceExplained.textContent = `${stats.variance_explained}%`;
        } else {
            varianceExplained.textContent = 'N/A';
        }
        
        // Kích hoạt nút thống kê
        toggleStatsBtn.disabled = false;
    }

    // Xử lý ảnh với API
    async function processImage(imageData, algorithm, mode, loader) {
        loader.classList.remove('hidden');
        
        try {
            const formData = new FormData();
            const blob = await fetch(imageData).then(res => res.blob());
            formData.append('image', blob, 'image.jpg');
            formData.append('algorithm', algorithm);
            formData.append('mode', mode);

            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            
            return result;
        } catch (error) {
            console.error('Error processing image:', error);
            alert(`Lỗi xử lý ảnh: ${error.message}`);
            throw error;
        } finally {
            loader.classList.add('hidden');
        }
    }

    // Lưu ảnh
    function saveImage(dataUrl, filename) {
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Chọn thuật toán LDA
    selectLdaBtn.addEventListener('click', () => {
        // Reset trạng thái nút
        selectPcaBtn.classList.remove('shadow-lg', 'bg-purple-700', 'ring-4', 'ring-purple-700', 'font-bold');
        selectPcaBtn.classList.add('bg-purple-500', 'hover:bg-purple-600');
        
        // Thiết lập style cho nút LDA
        selectLdaBtn.classList.remove('bg-green-500', 'hover:bg-green-600');
        selectLdaBtn.classList.add('shadow-lg', 'bg-green-700', 'ring-4', 'ring-green-700', 'font-bold');
        
        // Cập nhật thuật toán hiện tại
        currentAlgorithm = 'lda';
        
        // Kích hoạt nút nén
        compressBtn.disabled = false;
        decompressBtn.disabled = true;
    });

    // Chọn thuật toán PCA
    selectPcaBtn.addEventListener('click', () => {
        // Reset trạng thái nút
        selectLdaBtn.classList.remove('shadow-lg', 'bg-green-700', 'ring-4', 'ring-green-700', 'font-bold');
        selectLdaBtn.classList.add('bg-green-500', 'hover:bg-green-600');
        
        // Thiết lập style cho nút PCA
        selectPcaBtn.classList.remove('bg-purple-500', 'hover:bg-purple-600');
        selectPcaBtn.classList.add('shadow-lg', 'bg-purple-700', 'ring-4', 'ring-purple-700', 'font-bold');
        
        // Cập nhật thuật toán hiện tại
        currentAlgorithm = 'pca';
        
        // Kích hoạt nút nén
        compressBtn.disabled = false;
        decompressBtn.disabled = true;
    });

    // Nén ảnh
    compressBtn.addEventListener('click', async () => {
        if (!selectedImageData || !currentAlgorithm) {
            alert('Vui lòng chọn ảnh và phương pháp nén');
            return;
        }
        
        try {
            const result = await processImage(selectedImageData, currentAlgorithm, 'compress', compressLoader);
            
            // Hiển thị ảnh đã nén
            compressedImageData = result.image;
            compressedImage.src = compressedImageData;
            
            // Cập nhật thống kê
            updateStats(result.stats);
            
            // Kích hoạt các nút liên quan
            decompressBtn.disabled = false;
            saveCompressedBtn.disabled = false;
            
        } catch (error) {
            // Lỗi đã được xử lý trong hàm processImage
        }
    });

    // Giải nén ảnh
    decompressBtn.addEventListener('click', async () => {
        if (!currentAlgorithm) {
            alert('Vui lòng chọn phương pháp nén');
            return;
        }
        
        try {
            const result = await processImage(compressedImageData, currentAlgorithm, 'decompress', decompressLoader);
            
            // Hiển thị ảnh đã giải nén
            decompressedImageData = result.image;
            decompressedImage.src = decompressedImageData;
            
            // Cập nhật thống kê nếu có
            if (result.stats) {
                updateStats({...currentStats, ...result.stats});
            }
            
            // Kích hoạt nút lưu ảnh giải nén
            saveDecompressedBtn.disabled = false;
            
        } catch (error) {
            // Lỗi đã được xử lý trong hàm processImage
        }
    });

    // Hiển thị/ẩn bảng thống kê
    toggleStatsBtn.addEventListener('click', () => {
        statsPanel.classList.toggle('hidden');
    });

    // Lưu ảnh đã nén
    saveCompressedBtn.addEventListener('click', () => {
        if (compressedImageData) {
            saveImage(compressedImageData, `compressed_${currentAlgorithm}_${Date.now()}.jpg`);
        }
    });

    // Lưu ảnh đã giải nén
    saveDecompressedBtn.addEventListener('click', () => {
        if (decompressedImageData) {
            saveImage(decompressedImageData, `decompressed_${currentAlgorithm}_${Date.now()}.jpg`);
        }
    });

    // Kiểm tra trạng thái xử lý từ server
    async function checkProcessingStatus() {
        if (!currentAlgorithm) return;
        
        try {
            const mode = compressedImageData ? 'decompress' : 'compress';
            const response = await fetch(`http://localhost:5000/status/${currentAlgorithm}/${mode}`);
            const status = await response.json();
            
            // Cập nhật UI dựa trên trạng thái
            if (status.status === 'processing') {
                // Hiển thị loader và cập nhật tiến trình nếu có
                if (mode === 'compress') {
                    compressLoader.classList.remove('hidden');
                } else {
                    decompressLoader.classList.remove('hidden');
                }
            } else if (status.status === 'completed') {
                // Ẩn loader khi hoàn thành
                compressLoader.classList.add('hidden');
                decompressLoader.classList.add('hidden');
            }
        } catch (error) {
            console.log('Error checking status:', error);
        }
    }

    // Kiểm tra trạng thái định kỳ (mỗi 2 giây)
    setInterval(checkProcessingStatus, 2000);
});