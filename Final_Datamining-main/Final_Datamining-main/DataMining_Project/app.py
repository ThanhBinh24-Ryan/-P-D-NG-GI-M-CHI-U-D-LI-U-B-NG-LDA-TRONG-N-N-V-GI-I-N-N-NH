

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from lda_process import (
    nen_anh_bang_lda, giai_nen_anh_lda, 
    nen_anh_bang_pca, giai_nen_anh_pca,
    get_compression_stats
)
import os
import base64
import time
import shutil
import uuid

app = Flask(__name__, static_folder='static')
CORS(app)

# Cấu hình thư mục lưu trữ
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
COMPRESSED_DATA_FOLDER = "compressed_data"
LDA_DATA_FOLDER = os.path.join(COMPRESSED_DATA_FOLDER, "lda")
PCA_DATA_FOLDER = os.path.join(COMPRESSED_DATA_FOLDER, "pca")
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, LDA_DATA_FOLDER, PCA_DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Các biến theo dõi trạng thái xử lý
processing_status = {
    'lda_compress': {'status': 'idle', 'progress': 0},
    'lda_decompress': {'status': 'idle', 'progress': 0},
    'pca_compress': {'status': 'idle', 'progress': 0},
    'pca_decompress': {'status': 'idle', 'progress': 0}
}

# Route phục vụ các file tĩnh
@app.route('/', methods=['GET'])
def serve_index():
    return send_file('index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/static/<path:path>', methods=['GET'])
def serve_static_folder(path):
    return send_from_directory('static', path)

@app.route('/favicon.ico', methods=['GET'])
def serve_favicon():
    return '', 204

# Hàm tiện ích để chuyển đổi ảnh -> base64
def image_to_base64(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(image_path, 'rb') as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

# Hàm di chuyển file nén
def move_compressed_files(algorithm, files):
    target_folder = LDA_DATA_FOLDER if algorithm == 'lda' else PCA_DATA_FOLDER
    for src_path, filename in files:
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Compressed file not found: {src_path}")
        dest_path = os.path.join(target_folder, filename)
        shutil.copy(src_path, dest_path)

# API xử lý upload và xử lý ảnh
@app.route('/upload', methods=['POST'])
def upload_image():
    algorithm = request.form.get('algorithm')
    mode = request.form.get('mode')
    
    if algorithm not in ['lda', 'pca']:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    if mode not in ['compress', 'decompress']:
        return jsonify({'error': 'Invalid mode'}), 400
    
    # Tạo ID duy nhất cho mỗi yêu cầu để tránh xung đột file
    request_id = str(uuid.uuid4())
    status_key = f"{algorithm}_{mode}"
    
    try:
        # Cập nhật trạng thái
        processing_status[status_key]['status'] = 'processing'
        processing_status[status_key]['progress'] = 0
        
        if mode == 'compress':
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided for compression'}), 400
            
            image_file = request.files['image']
            # Sử dụng request_id để đảm bảo tên file duy nhất
            input_path = os.path.join(UPLOAD_FOLDER, f"image_{request_id}.jpg")
            image_file.save(input_path)
            
            # Kiểm tra file ảnh
            if not os.path.exists(input_path):
                return jsonify({'error': f'Failed to save image at {input_path}'}), 500
            
            # Lấy và xác thực tham số
            try:
                lda_components = int(request.form.get('lda_components', '1'))
                pca_components = int(request.form.get('pca_components', '50'))
                patch_size = int(request.form.get('patch_size', '5'))
                if lda_components < 1 or pca_components < 1 or patch_size < 1:
                    raise ValueError("Components and patch size must be positive integers")
            except ValueError as e:
                return jsonify({'error': f'Invalid parameters: {str(e)}'}), 400
            
            # Log để debug
            print(f"DEBUG: Request ID={request_id}, Algorithm={algorithm}, Mode={mode}")
            print(f"DEBUG: Input path={input_path}")
            print(f"DEBUG: LDA components={lda_components}, PCA components={pca_components}, Patch size={patch_size}")
            
            # Xử lý nén ảnh
            compressed_files = []
            if algorithm == 'lda':
                print(f"DEBUG: Calling nen_anh_bang_lda({input_path}, {lda_components}, {patch_size})")
                output_path, _, stats = nen_anh_bang_lda(
                    input_path,
                    so_luong_thanh_phan=lda_components,
                    kich_thuoc_cum=patch_size
                )
                compressed_files = [
                    ('ma_tran_nen_lda.npz', 'ma_tran_nen_lda.npz'),
                    ('ma_tran_nen_lda_W.npz', 'ma_tran_nen_lda_W.npz'),
                    ('kich_thuoc.npy', 'kich_thuoc.npy'),
                    ('ma_tran_shape_lda.npy', 'ma_tran_shape_lda.npy'),
                    ('kich_thuoc_cum_lda.npy', 'kich_thuoc_cum_lda.npy')
                ]
                if os.path.exists(output_path):
                    stats['compressed_picture'] = os.path.getsize(output_path)
                
                # Tính tổng kích thước các file nén
                total_compressed_size = 0
                for file_path, _ in compressed_files:
                    if os.path.exists(file_path):
                        total_compressed_size += os.path.getsize(file_path)
                
                stats['compressed_size'] = total_compressed_size
            else:  # pca
                print(f"DEBUG: Calling nen_anh_bang_pca({input_path}, {pca_components})")
                output_path, _, stats = nen_anh_bang_pca(
                    input_path,
                    so_luong_thanh_phan=pca_components
                )
                compressed_files = [
                    ('ma_tran_nen_pca.npz', 'ma_tran_nen_pca.npz'),
                    ('ma_tran_nen_pca_W.npz', 'ma_tran_nen_pca_W.npz'),
                    ('ma_tran_nen_pca_mean.npz', 'ma_tran_nen_pca_mean.npz'),
                    ('kich_thuoc.npy', 'kich_thuoc.npy'),
                    ('ma_tran_shape_pca.npy', 'ma_tran_shape_pca.npy')
                ]
            
            # Lưu file nén
            move_compressed_files(algorithm, compressed_files)
        
        else:  # decompress
            required_files = []
            if algorithm == 'lda':
                required_files = [
                    'ma_tran_nen_lda.npz',
                    'ma_tran_nen_lda_W.npz',
                    'kich_thuoc.npy',
                    'ma_tran_shape_lda.npy',
                    'kich_thuoc_cum_lda.npy'
                ]
            else:  # pca
                required_files = [
                    'ma_tran_nen_pca.npz',
                    'ma_tran_nen_pca_W.npz',
                    'ma_tran_nen_pca_mean.npz',
                    'kich_thuoc.npy',
                    'ma_tran_shape_pca.npy'
                ]
            
            # Kiểm tra file cần thiết
            data_folder = LDA_DATA_FOLDER if algorithm == 'lda' else PCA_DATA_FOLDER
            for filename in required_files:
                file_path = os.path.join(data_folder, filename)
                if not os.path.exists(file_path):
                    return jsonify({'error': f'Missing {filename} for decompression. Please compress an image first.'}), 400
                # Sao chép file vào thư mục hiện tại
                shutil.copy(file_path, os.path.join('.', filename))
            
            # Xử lý giải nén
            print(f"DEBUG: Calling {'giai_nen_anh_lda' if algorithm == 'lda' else 'giai_nen_anh_pca'}")
            if algorithm == 'lda':
                output_path, stats = giai_nen_anh_lda()
            else:
                output_path, stats = giai_nen_anh_pca()
            
            if output_path is None:
                return jsonify({'error': stats.get('error', 'Decompression failed')}), 500
        
        # Chuyển đổi ảnh kết quả sang base64
        print(f"DEBUG: Output path={output_path}")
        base64_image = image_to_base64(output_path)
        
        # Hoàn thành trạng thái
        processing_status[status_key]['status'] = 'completed'
        processing_status[status_key]['progress'] = 100
        
        return jsonify({
            'image': base64_image,
            'stats': stats
        })
        
    except Exception as e:
        print(f"DEBUG: Error={str(e)}")
        processing_status[status_key]['status'] = 'error'
        return jsonify({'error': str(e)}), 500
    finally:
        # Xóa file tạm
        if mode == 'compress' and 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        # Xóa các file nén tạm trong thư mục hiện tại
        temp_files = [
            'ma_tran_nen_lda.npz', 'ma_tran_nen_lda_W.npz', 'ma_tran_nen_lda_downsampled.npz',
            'ma_tran_nen_pca.npz', 'ma_tran_nen_pca_W.npz', 'ma_tran_nen_pca_mean.npz',
            'ma_tran_nen_pca_downsampled.npz', 'kich_thuoc.npy', 'ma_tran_shape_lda.npy',
            'ma_tran_shape_pca.npy', 'kich_thuoc_cum_lda.npy'
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# API kiểm tra trạng thái xử lý
@app.route('/status/<algorithm>/<mode>', methods=['GET'])
def check_status(algorithm, mode):
    status_key = f"{algorithm}_{mode}"
    if status_key in processing_status:
        return jsonify(processing_status[status_key])
    return jsonify({'error': 'Invalid status key'}), 400

# API lấy thông tin thống kê nén
@app.route('/stats/<algorithm>', methods=['GET'])
def get_stats(algorithm):
    if algorithm not in ['lda', 'pca']:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    stats = get_compression_stats(algorithm)
    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=True)