"""
Script demo nhanh để test các module
"""

def test_imports():
    """Test import các module"""
    print("🧪 Kiểm tra imports...")
    
    try:
        from nlp_toolkit import (
            DataCleaner, TextParser, BagOfWordsEncoder, 
            TfidfEncoder, Word2VecEncoder, CharNgramEncoder
        )
        print("✅ nlp_toolkit import OK")
    except Exception as e:
        print(f"❌ nlp_toolkit import FAILED: {e}")
        return False
    
    try:
        from ml_engine import MLTrainer, PerformanceEvaluator, ExperimentLogger
        print("✅ ml_engine import OK")
    except Exception as e:
        print(f"❌ ml_engine import FAILED: {e}")
        return False
    
    return True


def test_text_cleaning():
    """Test text cleaning"""
    print("\n🧪 Kiểm tra text cleaning...")
    
    from nlp_toolkit import DataCleaner
    
    test_text = "Hello WORLD!!! Visit https://example.com for more info... 12345"
    cleaned = DataCleaner.sanitize_content(test_text)
    
    print(f"Original: {test_text}")
    print(f"Cleaned:  {cleaned}")
    print("✅ Text cleaning OK")


def test_tokenization():
    """Test tokenization"""
    print("\n🧪 Kiểm tra tokenization...")
    
    from nlp_toolkit import TextParser
    
    text = "this is a test sentence"
    tokens = TextParser.extract_tokens(text)
    count = TextParser.get_token_count(text)
    
    print(f"Text:   {text}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {count}")
    print("✅ Tokenization OK")


def test_encoders():
    """Test các encoder"""
    print("\n🧪 Kiểm tra encoders...")
    
    from nlp_toolkit import BagOfWordsEncoder, TfidfEncoder, CharNgramEncoder
    
    corpus = [
        "this is a test",
        "another test document",
        "third document here"
    ]
    
    # Test BoW
    bow = BagOfWordsEncoder(max_vocab=50)
    bow_features = bow.fit_transform(corpus)
    print(f"✅ BoW: shape={bow_features.shape}, name={bow.get_name()}")
    
    # Test TF-IDF
    tfidf = TfidfEncoder(max_vocab=50)
    tfidf_features = tfidf.fit_transform(corpus)
    print(f"✅ TF-IDF: shape={tfidf_features.shape}, name={tfidf.get_name()}")
    
    # Test CharNgram
    char_ngram = CharNgramEncoder(max_features=100, ngram_range=(2, 3))
    char_features = char_ngram.fit_transform(corpus)
    print(f"✅ CharNgram: shape={char_features.shape}, name={char_ngram.get_name()}")


def test_ml_trainer():
    """Test ML trainer"""
    print("\n🧪 Kiểm tra ML trainer...")
    
    from ml_engine import MLTrainer
    
    trainer = MLTrainer()
    print(f"✅ MLTrainer khởi tạo thành công với {len(trainer.algorithms)} thuật toán")
    
    for algo_name in trainer.algorithms.keys():
        print(f"   - {algo_name}")


def main():
    """Chạy tất cả các test"""
    print("=" * 70)
    print("🚀 DEMO & TEST NLP TOOLKIT")
    print("=" * 70)
    
    if not test_imports():
        print("\n❌ Import failed! Kiểm tra lại dependencies.")
        return
    
    test_text_cleaning()
    test_tokenization()
    test_encoders()
    test_ml_trainer()
    
    print("\n" + "=" * 70)
    print("✅ TẤT CẢ TEST HOÀN TẤT!")
    print("=" * 70)
    print("\n💡 Bây giờ bạn có thể:")
    print("   1. Chạy: python convert_data.py (nếu có file JSON)")
    print("   2. Mở: jupyter notebook training_pipeline.ipynb")
    print("   3. Hoặc import và sử dụng trong code của bạn")


if __name__ == "__main__":
    main()
