import numpy as np
from PIL import Image
import os
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
import time
import json

# Các hằng số cấu hình
LDA_COMPONENTS = 5  # Số thành phần cho LDA
PCA_COMPONENTS = 50   # Số thành phần cho PCA
DOWNSAMPLE_FACTOR = 4  # Hệ số giảm kích thước
KMEANS_CLUSTERS = 10   # Số cụm K-means
PATCH_SIZE = 3 
# Thêm hàm mới để chia ảnh thành các cụm pixel
def chia_thanh_cum_pixel(ma_tran, kich_thuoc_cum=3):
    """Chia ma trận ảnh thành các cụm pixel kích thước NxN"""
    h, w = ma_tran.shape
    
    # Đảm bảo kích thước ma trận chia hết cho kích thước cụm
    h_padding = kich_thuoc_cum - (h % kich_thuoc_cum) if h % kich_thuoc_cum != 0 else 0
    w_padding = kich_thuoc_cum - (w % kich_thuoc_cum) if w % kich_thuoc_cum != 0 else 0
    
    if h_padding > 0 or w_padding > 0:
        # Thêm padding nếu cần
        padded = np.pad(ma_tran, ((0, h_padding), (0, w_padding)), mode='reflect')
        h, w = padded.shape
        ma_tran = padded
    
    # Tính số cụm theo mỗi chiều
    h_cum = h // kich_thuoc_cum
    w_cum = w // kich_thuoc_cum
    
    # Khởi tạo mảng để lưu các cụm
    cum_list = []
    
    # Trích xuất từng cụm
    for i in range(h_cum):
        for j in range(w_cum):
            h_start = i * kich_thuoc_cum
            h_end = (i + 1) * kich_thuoc_cum
            w_start = j * kich_thuoc_cum
            w_end = (j + 1) * kich_thuoc_cum
            
            cum = ma_tran[h_start:h_end, w_start:w_end].flatten()
            cum_list.append(cum)
    
    return np.array(cum_list), (h, w)
# Hàm chuyển ảnh thành ma trận và ngược lại
def anh_sang_ma_tran(duong_dan_anh, max_size=512):
    """Chuyển ảnh thành ma trận với khả năng thay đổi kích thước tối đa"""
    anh = Image.open(duong_dan_anh).convert('RGB')
    original_size = anh.size
    
    # Thay đổi kích thước nếu ảnh quá lớn để tăng hiệu suất
    if max(original_size) > max_size:
        ratio = max_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        anh = anh.resize(new_size, Image.LANCZOS)
    
    ma_tran = np.array(anh)
    return ma_tran, original_size

def ma_tran_sang_anh(ma_tran, kich_thuoc, duong_dan_luu, chat_luong=80):
    """Chuyển ma trận thành ảnh với tùy chọn chất lượng"""
    ma_tran = np.clip(ma_tran, 0, 255)
    anh = Image.fromarray(ma_tran.astype(np.uint8))
    anh = anh.resize(kich_thuoc, Image.LANCZOS)
    anh.save(duong_dan_luu, 'JPEG', quality=chat_luong)
    return duong_dan_luu

# LDA thủ công với tối ưu hóa
def lda_thu_cong(X, y, so_luong_thanh_phan):
    """Triển khai LDA thủ công đã được tối ưu"""
    classes = np.unique(y)
    n_samples, n_features = X.shape
    
    # Tính toán các vector trung bình trên các lớp
    mean_vectors = []
    for cl in classes:
        class_data = X[y == cl]
        if len(class_data) > 0:  # Đảm bảo có dữ liệu cho lớp
            mean_vectors.append(np.mean(class_data, axis=0))
    mean_vectors = np.array(mean_vectors)
     
    # Tính ma trận phân tán trong lớp (Sw)
    Sw = np.zeros((n_features, n_features))
    for cl, mean_vec in zip(classes, mean_vectors):
        class_data = X[y == cl]
        if len(class_data) > 0:
            class_centered = class_data - mean_vec
            Sw += class_centered.T.dot(class_centered)
    
    # Tính ma trận phân tán giữa các lớp (Sb)
    overall_mean = np.mean(X, axis=0)
    Sb = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == classes[i]].shape[0]
        if n > 0:
            mean_diff = mean_vec - overall_mean
            Sb += n * np.outer(mean_diff, mean_diff)
    
    # Giải quyết vấn đề Sw có thể không khả nghịch
    try:
        # Thêm một ít nhiễu để đảm bảo ma trận Sw là khả nghịch
        Sw_reg = Sw + np.eye(n_features) * 1e-4
        Sw_inv = np.linalg.inv(Sw_reg)
        eigvals, eigvecs = np.linalg.eigh(Sw_inv.dot(Sb))
    except np.linalg.LinAlgError:
        # Sử dụng pinv nếu không thể đảo ngược ma trận
        Sw_reg = Sw + np.eye(n_features) * 1e-3
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw_reg).dot(Sb))
    
    # Sắp xếp eigenvalues và eigenvectors theo thứ tự giảm dần
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Lấy các eigenvectors tương ứng với các eigenvalues lớn nhất
    W = eigvecs[:, :so_luong_thanh_phan]
    
    # Chiếu dữ liệu lên không gian LDA
    X_lda = X.dot(W)
    
    return X_lda, W

# PCA thủ công với tối ưu hóa
def pca_thu_cong(X, so_luong_thanh_phan):
    """Triển khai PCA thủ công đã được tối ưu"""
    # Trung tâm hóa dữ liệu
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Tính ma trận hiệp phương sai
    n_samples = X.shape[0]
    if n_samples > X.shape[1]:
        # Phương pháp truyền thống
        cov_matrix = np.cov(X_centered.T)
    else:
        # Tối ưu tính toán cho ma trận cao chiều
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Tính eigenvalues và eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    except np.linalg.LinAlgError:
        # Thêm nhiễu nếu ma trận không ổn định
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-5
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Sắp xếp theo thứ tự giảm dần
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Lấy các eigenvectors hàng đầu
    W = eigvecs[:, :so_luong_thanh_phan]
    
    # Chiếu dữ liệu vào không gian thấp hơn
    X_pca = X_centered.dot(W)
    
    # Tính toán tỷ lệ giải thích phương sai
    explained_variance = eigvals[:so_luong_thanh_phan] / np.sum(eigvals)
    total_variance = np.sum(explained_variance)
    
    return X_pca, W, X_mean, total_variance

# Hàm giảm độ phân giải được cải thiện
def downsample_matrix(matrix, factor=DOWNSAMPLE_FACTOR):
    """Giảm kích thước ma trận với hệ số giảm định trước"""
    h, w = matrix.shape
    new_h, new_w = h // factor, w // factor
    
    # Sử dụng scipy zoom cho phép tái tạo hình ảnh tốt hơn
    downsampled = zoom(matrix, (new_h/h, new_w/w), order=1)
    return downsampled

# Hàm khôi phục độ phân giải được cải thiện
def upsample_matrix(matrix, original_shape, factor=DOWNSAMPLE_FACTOR):
    """Tăng kích thước ma trận về kích thước ban đầu"""
    h, w = original_shape
    return zoom(matrix, (h/matrix.shape[0], w/matrix.shape[1]), order=3)

def tai_tao_tu_cum_pixel(cum_list, kich_thuoc_ma_tran, kich_thuoc_cum=3):
    """Tái tạo ma trận từ danh sách các cụm pixel"""
    h, w = kich_thuoc_ma_tran
    ma_tran = np.zeros((h, w))
    
    h_cum = h // kich_thuoc_cum
    w_cum = w // kich_thuoc_cum
    
    cum_index = 0
    for i in range(h_cum):
        for j in range(w_cum):
            h_start = i * kich_thuoc_cum
            h_end = (i + 1) * kich_thuoc_cum
            w_start = j * kich_thuoc_cum
            w_end = (j + 1) * kich_thuoc_cum
            
            ma_tran[h_start:h_end, w_start:w_end] = cum_list[cum_index].reshape(kich_thuoc_cum, kich_thuoc_cum)
            cum_index += 1
            
    return ma_tran

# Hàm nén ảnh bằng LDA
def nen_anh_bang_lda(duong_dan_anh, so_luong_thanh_phan=LDA_COMPONENTS, kich_thuoc_cum=PATCH_SIZE):
    """Nén ảnh sử dụng thuật toán LDA với phân cụm pixel"""
    start_time = time.time()
    
    # Đọc và xử lý ảnh
    ma_tran, kich_thuoc = anh_sang_ma_tran(duong_dan_anh)
    h, w, _ = ma_tran.shape
    
    # Thông tin nén
    original_size = os.path.getsize(duong_dan_anh)
    compression_stats = {
        "original_size": original_size,
        "original_dimensions": [kich_thuoc[0], kich_thuoc[1]],
        "processed_dimensions": [h, w],
        "algorithm": "LDA",
        "components": so_luong_thanh_phan,
        "patch_size": kich_thuoc_cum
    }
    
    # Xử lý từng kênh màu
    ma_tran_nen_list = []
    ma_tran_downsampled_list = []
    W_list = []
    
    for channel in range(3):
        ma_tran_channel = ma_tran[:, :, channel]
        
        # Chia thành các cụm pixel
        cum_pixel_list, kich_thuoc_ma_tran = chia_thanh_cum_pixel(ma_tran_channel, kich_thuoc_cum)
        
        # Phân cụm các cụm pixel bằng K-means
        kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10).fit(cum_pixel_list)
        nhan_gia = kmeans.labels_
        
        # Áp dụng LDA
        ma_tran_nen_channel, W = lda_thu_cong(cum_pixel_list, nhan_gia, so_luong_thanh_phan)
        
        # Tái tạo dữ liệu
        ma_tran_reconstructed_cum = ma_tran_nen_channel.dot(W.T)
        ma_tran_reconstructed = tai_tao_tu_cum_pixel(
            ma_tran_reconstructed_cum, 
            kich_thuoc_ma_tran, 
            kich_thuoc_cum
        )
        
        # Giảm kích thước để tiết kiệm bộ nhớ
        ma_tran_downsampled = downsample_matrix(ma_tran_reconstructed)
        
        # Lưu kết quả với độ chính xác giảm để tiết kiệm không gian
        ma_tran_nen_list.append(ma_tran_nen_channel.astype(np.float16))
        ma_tran_downsampled_list.append(ma_tran_downsampled)
        W_list.append(W.astype(np.float16))
    
    # Lưu dữ liệu nén và thông tin kích thước
    np.savez_compressed('ma_tran_nen_lda.npz', ma_tran_nen_list)
    np.savez_compressed('ma_tran_nen_lda_downsampled.npz', ma_tran_downsampled_list) 
    np.savez_compressed('ma_tran_nen_lda_W.npz', W_list)
    
    # Lưu kích thước và hình dạng ma trận
    np.save('kich_thuoc.npy', kich_thuoc)
    np.save('ma_tran_shape_lda.npy', np.array([h, w]))
    
    # Lưu thông tin kích thước cụm
    np.save('kich_thuoc_cum_lda.npy', kich_thuoc_cum)
    
    # Tạo và lưu ảnh nén
    ma_tran_tam = np.zeros((h, w, 3))
    for channel in range(3):
        ma_tran_tam[:, :, channel] = upsample_matrix(ma_tran_downsampled_list[channel], (h, w))
    
    duong_dan_nen = 'results/anh_nen_lda.jpg'
    ma_tran_sang_anh(ma_tran_tam, kich_thuoc, duong_dan_nen)
    
    # Tính thống kê nén
    compressed_picture = os.path.getsize(duong_dan_nen)
    compressed_size = os.path.getsize('ma_tran_nen_lda.npz') + os.path.getsize('ma_tran_nen_lda_W.npz') + os.path.getsize('kich_thuoc.npy')
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    processing_time = time.time() - start_time
    
    compression_stats.update({
        "compressed_size": compressed_size,
        "conpressed_picture": compressed_picture,
        "compression_ratio": round(compression_ratio, 2),
        "processing_time": round(processing_time, 3)
    })
    
    # Lưu thông tin thống kê
    with open('lda_stats.json', 'w') as f:
        json.dump(compression_stats, f)
    
    return duong_dan_nen, kich_thuoc, compression_stats

# Hàm giải nén ảnh từ LDA
def giai_nen_anh_lda():
    """Giải nén ảnh từ dữ liệu đã được nén bằng LDA"""
    start_time = time.time()
    
    try:
        # Đọc dữ liệu đã nén
        ma_tran_nen_list = np.load('ma_tran_nen_lda.npz', allow_pickle=True)['arr_0']
        W_list = np.load('ma_tran_nen_lda_W.npz', allow_pickle=True)['arr_0']
        kich_thuoc = np.load('kich_thuoc.npy', allow_pickle=True)
        
        # Đọc hình dạng ma trận và kích thước cụm nếu có
        if os.path.exists('ma_tran_shape_lda.npy'):
            h, w = np.load('ma_tran_shape_lda.npy')
        else:
            # Nếu file không tồn tại, thử tính toán kích thước từ W_list
            w = W_list[0].shape[0]
            # Lấy số lượng thành phần từ kích thước của ma trận W
            so_luong_thanh_phan_lda = W_list[0].shape[1]
            h = len(ma_tran_nen_list[0]) * so_luong_thanh_phan_lda // w
        
        # Đọc kích thước cụm
        if os.path.exists('kich_thuoc_cum_lda.npy'):
            kich_thuoc_cum = np.load('kich_thuoc_cum_lda.npy').item()
        else:
            # Giá trị mặc định
            kich_thuoc_cum = PATCH_SIZE
        
        # Khôi phục ảnh từ dữ liệu nén
        ma_tran_giai_nen = np.zeros((h, w, 3))
        
        for channel in range(3):
            # Tái tạo các cụm pixel
            ma_tran_reconstructed_cum = ma_tran_nen_list[channel].dot(W_list[channel].T)
            
            # Tính toán số cụm theo mỗi chiều
            h_padded = ((h + kich_thuoc_cum - 1) // kich_thuoc_cum) * kich_thuoc_cum
            w_padded = ((w + kich_thuoc_cum - 1) // kich_thuoc_cum) * kich_thuoc_cum
            
            # Tái tạo ma trận từ các cụm pixel
            ma_tran_channel = tai_tao_tu_cum_pixel(
                ma_tran_reconstructed_cum, 
                (h_padded, w_padded), 
                kich_thuoc_cum
            )
            
            # Cắt về kích thước ban đầu
            ma_tran_giai_nen[:, :, channel] = ma_tran_channel[:h, :w]
        
        # Chuyển kích thước từ (width, height) thành (width, height) cho ma_tran_sang_anh
        w_out, h_out = kich_thuoc
        
        duong_dan_giai_nen = 'results/anh_giai_nen_lda.jpg'
        ma_tran_sang_anh(ma_tran_giai_nen, (w_out, h_out), duong_dan_giai_nen)
        
        # Đọc thông tin thống kê nếu có
        stats = {}
        if os.path.exists('lda_stats.json'):
            with open('lda_stats.json', 'r') as f:
                stats = json.load(f)
        
        # Cập nhật thông tin giải nén
        stats["decompression_time"] = round(time.time() - start_time, 3)
        
        # Lưu lại thông tin
        with open('lda_stats.json', 'w') as f:
            json.dump(stats, f)
        
        return duong_dan_giai_nen, stats
    
    except Exception as e:
        print(f"Lỗi giải nén ảnh LDA: {str(e)}")
        # Trả về thông tin lỗi chi tiết
        return None, {"error": str(e)}

def giai_nen_anh_pca():
    """Giải nén ảnh từ dữ liệu đã được nén bằng PCA"""
    start_time = time.time()
    
    try:
        # Đọc dữ liệu đã nén
        ma_tran_nen_list = np.load('ma_tran_nen_pca.npz', allow_pickle=True)['arr_0']
        W_list = np.load('ma_tran_nen_pca_W.npz', allow_pickle=True)['arr_0']
        mean_list = np.load('ma_tran_nen_pca_mean.npz', allow_pickle=True)['arr_0']
        kich_thuoc = np.load('kich_thuoc.npy', allow_pickle=True)
        
        # Đọc hình dạng ma trận nếu có
        if os.path.exists('ma_tran_shape_pca.npy'):
            h, w = np.load('ma_tran_shape_pca.npy')
        else:
            # Nếu file không tồn tại, thử tính toán kích thước từ W_list
            w = W_list[0].shape[0]
            # Lấy số lượng thành phần từ kích thước của ma trận W hoặc sử dụng giá trị mặc định 
            so_luong_thanh_phan_pca = W_list[0].shape[1]
            h = len(ma_tran_nen_list[0]) * so_luong_thanh_phan_pca // w
        
        # Kiểm tra kích thước và điều chỉnh nếu cần
        expected_size = h * w
        actual_size = ma_tran_nen_list[0].shape[0] * W_list[0].shape[1]
        
        if actual_size != expected_size:
            # Tính toán lại chiều cao dựa trên kích thước thực tế
            w_orig = W_list[0].shape[0]
            h_orig = ma_tran_nen_list[0].dot(W_list[0].T).size // w_orig
            h, w = h_orig, w_orig
        
        # Khôi phục ảnh từ dữ liệu nén
        ma_tran_giai_nen = np.zeros((h, w, 3))
        for channel in range(3):
            reconstructed = ma_tran_nen_list[channel].dot(W_list[channel].T) + mean_list[channel]
            ma_tran_giai_nen[:, :, channel] = reconstructed.reshape(h, w)
        
        # Chuyển kích thước từ (width, height) thành (width, height) cho ma_tran_sang_anh
        w_out, h_out = kich_thuoc
        
        duong_dan_giai_nen = 'results/anh_giai_nen_pca.jpg'
        ma_tran_sang_anh(ma_tran_giai_nen, (w_out, h_out), duong_dan_giai_nen)
        
        # Đọc thông tin thống kê nếu có
        stats = {}
        if os.path.exists('pca_stats.json'):
            with open('pca_stats.json', 'r') as f:
                stats = json.load(f)
        
        # Cập nhật thông tin giải nén
        stats["decompression_time"] = round(time.time() - start_time, 3)
        
        # Lưu lại thông tin
        with open('pca_stats.json', 'w') as f:
            json.dump(stats, f)
        
        return duong_dan_giai_nen, stats
    
    except Exception as e:
        print(f"Lỗi giải nén ảnh PCA: {str(e)}")
        # Trả về thông tin lỗi chi tiết
        return None, {"error": str(e)}
# Hàm nén ảnh bằng PCA
def nen_anh_bang_pca(duong_dan_anh, so_luong_thanh_phan=PCA_COMPONENTS):
    """Nén ảnh sử dụng thuật toán PCA"""
    start_time = time.time()
    
    # Đọc và xử lý ảnh
    ma_tran, kich_thuoc = anh_sang_ma_tran(duong_dan_anh)
    h, w, _ = ma_tran.shape
    
    # Thông tin nén
    original_size = os.path.getsize(duong_dan_anh)
    compression_stats = {
        "original_size": original_size,
        "original_dimensions": [kich_thuoc[0], kich_thuoc[1]],
        "processed_dimensions": [h, w],
        "algorithm": "PCA",
        "components": so_luong_thanh_phan
    }
    
    # Xử lý từng kênh màu
    ma_tran_nen_list = []
    ma_tran_downsampled_list = []
    W_list = []
    mean_list = []
    total_variance_explained = 0
    
    for channel in range(3):
        ma_tran_channel = ma_tran[:, :, channel].reshape(-1, w)
        
        # Áp dụng PCA
        ma_tran_nen_channel, W, mean, variance_explained = pca_thu_cong(ma_tran_channel, so_luong_thanh_phan)
        ma_tran_reconstructed = (ma_tran_nen_channel.dot(W.T) + mean).reshape(h, w)
        
        # Lưu tỷ lệ phương sai giải thích cho từng kênh
        total_variance_explained += variance_explained / 3
        
        # Giảm kích thước để tiết kiệm bộ nhớ
        ma_tran_downsampled = downsample_matrix(ma_tran_reconstructed)
        
        # Lưu kết quả với độ chính xác giảm để tiết kiệm không gian
        ma_tran_nen_list.append(ma_tran_nen_channel.astype(np.float16))
        ma_tran_downsampled_list.append(ma_tran_downsampled)
        W_list.append(W.astype(np.float16))
        mean_list.append(mean.astype(np.float16))
    
    # Lưu dữ liệu nén và thông tin kích thước
    np.savez_compressed('ma_tran_nen_pca.npz', ma_tran_nen_list)
    np.savez_compressed('ma_tran_nen_pca_downsampled.npz', ma_tran_downsampled_list)
    np.savez_compressed('ma_tran_nen_pca_W.npz', W_list)
    np.savez_compressed('ma_tran_nen_pca_mean.npz', mean_list)
    np.save('kich_thuoc.npy', kich_thuoc)
    np.save('ma_tran_shape_pca.npy', np.array([h, w]))
    
    # Tạo và lưu ảnh nén
    ma_tran_tam = np.zeros((h, w, 3))
    for channel in range(3):
        ma_tran_tam[:, :, channel] = upsample_matrix(ma_tran_downsampled_list[channel], (h, w))
    
    duong_dan_nen = 'results/anh_nen_pca.jpg'
    ma_tran_sang_anh(ma_tran_tam, kich_thuoc, duong_dan_nen)
    
    # Tính thống kê nén
    compressed_size = (
        os.path.getsize('ma_tran_nen_pca.npz') + 
        os.path.getsize('ma_tran_nen_pca_W.npz') + 
        os.path.getsize('ma_tran_nen_pca_mean.npz') + 
        os.path.getsize('kich_thuoc.npy')
    )
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    processing_time = time.time() - start_time
    
    compression_stats.update({
        "compressed_size": compressed_size,
        "compression_ratio": round(compression_ratio, 2),
        "variance_explained": round(total_variance_explained * 100, 2),
        "processing_time": round(processing_time, 3)
    })
    
    # Lưu thông tin thống kê
    with open('pca_stats.json', 'w') as f:
        json.dump(compression_stats, f)
    
    return duong_dan_nen, kich_thuoc, compression_stats

# Hàm giải nén ảnh từ PCA
def giai_nen_anh_pca():
    """Giải nén ảnh từ dữ liệu đã được nén bằng PCA"""
    start_time = time.time()
    
    try:
        # Đọc dữ liệu đã nén
        ma_tran_nen_list = np.load('ma_tran_nen_pca.npz', allow_pickle=True)['arr_0']
        W_list = np.load('ma_tran_nen_pca_W.npz', allow_pickle=True)['arr_0']
        mean_list = np.load('ma_tran_nen_pca_mean.npz', allow_pickle=True)['arr_0']
        kich_thuoc = np.load('kich_thuoc.npy', allow_pickle=True)
        
        # Đọc hình dạng ma trận nếu có
        if os.path.exists('ma_tran_shape_pca.npy'):
            h, w = np.load('ma_tran_shape_pca.npy')
        else:
            # Nếu file không tồn tại, thử tính toán kích thước từ W_list
            w = W_list[0].shape[0]
            h = len(ma_tran_nen_list[0]) * PCA_COMPONENTS // w
        
        # Kiểm tra kích thước và điều chỉnh nếu cần
        expected_size = h * w
        actual_size = ma_tran_nen_list[0].shape[0] * W_list[0].shape[1]
        
        if actual_size != expected_size:
            # Tính toán lại chiều cao dựa trên kích thước thực tế
            w_orig = W_list[0].shape[0]
            h_orig = ma_tran_nen_list[0].dot(W_list[0].T).size // w_orig
            h, w = h_orig, w_orig
        
        # Khôi phục ảnh từ dữ liệu nén
        ma_tran_giai_nen = np.zeros((h, w, 3))
        for channel in range(3):
            reconstructed = ma_tran_nen_list[channel].dot(W_list[channel].T) + mean_list[channel]
            ma_tran_giai_nen[:, :, channel] = reconstructed.reshape(h, w)
        
        # Chuyển kích thước từ (width, height) thành (width, height) cho ma_tran_sang_anh
        w_out, h_out = kich_thuoc
        
        duong_dan_giai_nen = 'results/anh_giai_nen_pca.jpg'
        ma_tran_sang_anh(ma_tran_giai_nen, (w_out, h_out), duong_dan_giai_nen)
        
        # Đọc thông tin thống kê nếu có
        stats = {}
        if os.path.exists('pca_stats.json'):
            with open('pca_stats.json', 'r') as f:
                stats = json.load(f)
        
        # Cập nhật thông tin giải nén
        stats["decompression_time"] = round(time.time() - start_time, 3)
        
        # Lưu lại thông tin
        with open('pca_stats.json', 'w') as f:
            json.dump(stats, f)
        
        return duong_dan_giai_nen, stats
    
    except Exception as e:
        print(f"Lỗi giải nén ảnh PCA: {str(e)}")
        # Trả về thông tin lỗi chi tiết
        return None, {"error": str(e)}


# Hàm lấy thông tin thống kê nén
def get_compression_stats(algorithm):
    """Lấy thông tin thống kê cho thuật toán đã chọn"""
    if algorithm == 'lda' and os.path.exists('lda_stats.json'):
        with open('lda_stats.json', 'r') as f:
            return json.load(f)
    elif algorithm == 'pca' and os.path.exists('pca_stats.json'):
        with open('pca_stats.json', 'r') as f:
            return json.load(f)
    return {}

