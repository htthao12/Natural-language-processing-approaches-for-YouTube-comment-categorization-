"""
Script chuyển đổi dữ liệu từ JSON sang Parquet
Chạy script này một lần để convert data format
"""

import json
import pandas as pd
import sys


def convert_data(json_path='youtube_.json', parquet_path='data.parquet'):
    """Chuyển đổi file JSON sang Parquet"""
    
    print("=" * 70)
    print("CHUYỂN ĐỔI DỮ LIỆU JSON -> PARQUET")
    print("=" * 70)
    
    try:
        # Đọc file JSON
        print(f"\n📖 Đang đọc file: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Kiểm tra cấu trúc dữ liệu
        if 'comments' in data:
            df = pd.DataFrame(data['comments'])
            print(f"✅ Tìm thấy {len(df)} comments")
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            print(f"✅ Tìm thấy {len(df)} records")
        else:
            df = pd.DataFrame([data])
            print(f"✅ Chuyển đổi single record")
        
        # Hiển thị thông tin DataFrame
        print(f"\n📊 Thông tin DataFrame:")
        print(f"  - Số hàng: {len(df)}")
        print(f"  - Số cột: {len(df.columns)}")
        print(f"  - Các cột: {list(df.columns)}")
        
        # Kiểm tra dữ liệu mẫu
        print(f"\n🔍 Dữ liệu mẫu (5 hàng đầu):")
        print(df.head())
        
        # Lưu thành file Parquet
        print(f"\n💾 Đang lưu vào file: {parquet_path}")
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
        # Kiểm tra file đã lưu
        df_check = pd.read_parquet(parquet_path)
        print(f"✅ Đã lưu thành công!")
        print(f"  - Kích thước file Parquet: {len(df_check)} records")
        
        print(f"\n{'=' * 70}")
        print("✅ CHUYỂN ĐỔI HOÀN TẤT!")
        print(f"{'=' * 70}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {json_path}")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi: {str(e)}")
        return False


if __name__ == "__main__":
    # Lấy tham số từ command line nếu có
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        parquet_file = sys.argv[2] if len(sys.argv) > 2 else 'data.parquet'
    else:
        json_file = 'youtube_.json'
        parquet_file = 'data.parquet'
    
    # Thực hiện chuyển đổi
    success = convert_data(json_file, parquet_file)
    
    if success:
        print(f"\n💡 Bây giờ bạn có thể sử dụng file '{parquet_file}' trong training pipeline!")
    else:
        print("\n❌ Chuyển đổi thất bại. Vui lòng kiểm tra lại file JSON.")
