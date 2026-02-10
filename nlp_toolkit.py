import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import scipy.sparse as sp
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import AutoTokenizer, AutoModel


class DataCleaner:
    
    @staticmethod
    def sanitize_content(raw_input):
        clean_text = raw_input.lower()
        clean_text = re.sub(r'https?://\S+|www\.\S+', '', clean_text)
        clean_text = re.sub(r'[^a-z0-9\s]', ' ', clean_text)
        clean_text = re.sub(r'\d+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    
    @staticmethod
    def remove_redundant_entries(data_list):
        unique_data = []
        seen_content = set()
        
        for item in data_list:
            cleaned = DataCleaner.sanitize_content(item['content'])
            if cleaned and cleaned not in seen_content:
                seen_content.add(cleaned)
                unique_data.append(item)
        
        return unique_data
    
    @staticmethod
    def validate_length(data_list, min_tokens=3):
        return [
            item for item in data_list
            if len(item['content'].split()) >= min_tokens
        ]


class TextParser:
    
    @staticmethod
    def extract_tokens(text_input):
        return text_input.split()
    
    @staticmethod
    def get_token_count(text_input):
        return len(TextParser.extract_tokens(text_input))


class DataLoader:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None
        self.training_samples = []
        self.inference_samples = []
    
    def process_dataset(self, sample_limit=None):
        print("=" * 70)
        print("BẮT ĐẦU XỬ LÝ DỮ LIỆU")
        print("=" * 70)
        
        df = pd.read_parquet(self.file_path)
        df = df.head(800)
        if sample_limit:
            df = df.head(sample_limit)
        
        print(f"Tổng số comment ban đầu: {len(df)}")
        
        self.dataset = df.to_dict('records')
        self.dataset = DataCleaner.remove_redundant_entries(self.dataset)
        print(f"Sau khi loại bỏ trùng lặp: {len(self.dataset)}")
        
        self.dataset = DataCleaner.validate_length(self.dataset)
        print(f"Sau khi lọc độ dài: {len(self.dataset)}")
        
        print("Đang chuẩn hóa văn bản:")
        for entry in tqdm(self.dataset, desc="Xử lý"):
            entry['cleaned_content'] = DataCleaner.sanitize_content(entry['content'])
        
        all_texts = [e['cleaned_content'] for e in self.dataset]
        all_labels = [e['label'] if e['label'] else 'unlabeled' for e in self.dataset]
        
        self.training_samples = [
            (text, label) for text, label in zip(all_texts, all_labels)
            if label != 'unlabeled' and text
        ]
        
        self.inference_samples = [
            (text, label) for text, label in zip(all_texts, all_labels)
            if label == 'unlabeled' and text
        ]
        
        print(f"Mẫu có label: {len(self.training_samples)}")
        print(f"Mẫu không có label: {len(self.inference_samples)}")
        
        return self.training_samples, self.inference_samples


class BagOfWordsEncoder:
    
    def __init__(self, max_vocab=1000, ngram_config=(1, 2), 
                 min_freq=2, max_freq=0.8, binary_mode=False):
        self.max_vocab = max_vocab
        self.ngram_config = ngram_config
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.binary_mode = binary_mode
        
        self.encoder = CountVectorizer(
            max_features=self.max_vocab,
            ngram_range=self.ngram_config,
            min_df=self.min_freq,
            max_df=self.max_freq,
            binary=self.binary_mode
        )
        
        self.name = "Binary BoW" if binary_mode else "Count BoW"
    
    def fit(self, corpus):
        self.encoder.fit(corpus)
        return self
    
    def transform(self, corpus):
        return self.encoder.transform(corpus)
    
    def fit_transform(self, corpus):
        return self.encoder.fit_transform(corpus)
    
    def get_feature_names(self):
        return self.encoder.get_feature_names_out()
    
    def get_name(self):
        return self.name


class TfidfEncoder:
    
    def __init__(self, max_vocab=1000, ngram_config=(1, 3),
                 min_freq=2, max_freq=0.8,
                 use_reduction=False, reduced_dims=100):
        
        self.max_vocab = max_vocab
        self.ngram_config = ngram_config
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.use_reduction = use_reduction
        self.reduced_dims = reduced_dims
        
        self.encoder = TfidfVectorizer(
            max_features=self.max_vocab,
            ngram_range=self.ngram_config,
            min_df=self.min_freq,
            max_df=self.max_freq,
            sublinear_tf=True
        )
        
        self.dimension_reducer = None
        if self.use_reduction:
            self.dimension_reducer = TruncatedSVD(
                n_components=self.reduced_dims,
                random_state=42
            )
            self.name = "TF-IDF + LSA"
        else:
            self.name = "TF-IDF"
    
    def fit(self, corpus):
        self.encoder.fit(corpus)
        
        if self.use_reduction:
            matrix = self.encoder.transform(corpus)
            n_comp = min(
                self.reduced_dims,
                matrix.shape[1] - 1,
                matrix.shape[0] - 1
            )
            self.dimension_reducer.n_components = n_comp
            self.dimension_reducer.fit(matrix)
        
        return self
    
    def transform(self, corpus):
        matrix = self.encoder.transform(corpus)
        
        if self.use_reduction:
            return self.dimension_reducer.transform(matrix)
        
        return matrix
    
    def fit_transform(self, corpus):
        """Huấn luyện và chuyển đổi"""
        matrix = self.encoder.fit_transform(corpus)
        
        if self.use_reduction:
            n_comp = min(
                self.reduced_dims,
                matrix.shape[1] - 1,
                matrix.shape[0] - 1
            )
            self.dimension_reducer.n_components = n_comp
            return self.dimension_reducer.fit_transform(matrix)
        
        return matrix
    
    def get_feature_names(self):
        """Lấy danh sách tên feature"""
        return self.encoder.get_feature_names_out()
    
    def get_name(self):
        """Trả về tên encoder"""
        return self.name


class Word2VecEncoder:
    
    def __init__(self, vector_size=100, window_size=5,
                 min_count=2, workers=4, epochs=10):
        
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        
        self.model = None
        self.name = "Word2Vec"
    
    def fit(self, corpus):
        """Huấn luyện Word2Vec model"""
        tokenized = [TextParser.extract_tokens(text) for text in corpus]
        
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        
        return self
    
    def transform(self, corpus):
        """Chuyển đổi corpus thành embeddings"""
        embeddings = []
        
        for text in corpus:
            tokens = TextParser.extract_tokens(text)
            vectors = []
            
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
            
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        
        return np.array(embeddings)
    
    def fit_transform(self, corpus):
        """Huấn luyện và chuyển đổi"""
        self.fit(corpus)
        return self.transform(corpus)
    
    def get_name(self):
        """Trả về tên encoder"""
        return self.name


class FastTextEncoder:
    
    def __init__(self, vector_size=100, window_size=5,
                 min_count=2, epochs=10):
        
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.epochs = epochs
        
        self.model = None
        self.name = "FastText-style"
    
    def fit(self, corpus):
        """Huấn luyện FastText-style model"""
        tokenized = [TextParser.extract_tokens(text) for text in corpus]
        
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window_size,
            min_count=self.min_count,
            sg=1,  # Skip-gram
            epochs=self.epochs,
            workers=4
        )
        
        return self
    
    def transform(self, corpus):
        """Chuyển đổi corpus thành embeddings"""
        embeddings = []
        
        for text in corpus:
            tokens = TextParser.extract_tokens(text)
            vectors = []
            
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
            
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        
        return np.array(embeddings)
    
    def fit_transform(self, corpus):
        """Huấn luyện và chuyển đổi"""
        self.fit(corpus)
        return self.transform(corpus)
    
    def get_name(self):
        """Trả về tên encoder"""
        return self.name


class CharNgramEncoder:
    
    def __init__(self, max_features=5000, ngram_range=(2, 5),
                 min_freq=2, max_freq=0.9):
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        self.encoder = TfidfVectorizer(
            max_features=self.max_features,
            analyzer='char',  # Character level
            ngram_range=self.ngram_range,
            min_df=self.min_freq,
            max_df=self.max_freq
        )
        
        self.name = f"Char-{ngram_range[0]}-{ngram_range[1]}gram"
    
    def fit(self, corpus):
        """Huấn luyện encoder"""
        self.encoder.fit(corpus)
        return self
    
    def transform(self, corpus):
        """Chuyển đổi corpus thành vector"""
        return self.encoder.transform(corpus)
    
    def fit_transform(self, corpus):
        """Huấn luyện và chuyển đổi"""
        return self.encoder.fit_transform(corpus)
    
    def get_feature_names(self):
        """Lấy danh sách tên feature"""
        return self.encoder.get_feature_names_out()
    
    def get_name(self):
        """Trả về tên encoder"""
        return self.name


class PhoBertEncoder:
    
    def __init__(self, model_name='vinai/phobert-base', max_length=256, 
                 pooling='mean', device=None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Đang load PhoBERT model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.name = f"PhoBERT-{pooling}"
        print(f"✅ Đã load PhoBERT trên {self.device}")
    
    def fit(self, corpus):
        return self
    
    def transform(self, corpus, batch_size=16):
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(corpus), batch_size), desc="BERT Encoding"):
                batch_texts = corpus[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                
                if self.pooling == 'cls':
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling == 'mean':
                    attention_mask = encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                else:
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def fit_transform(self, corpus, batch_size=16):
        return self.transform(corpus, batch_size)
    
    def get_name(self):
        return self.name


def convert_json_to_parquet(json_path, parquet_path):
    import json
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Giả sử structure là {'comments': [...]}
    if 'comments' in data:
        df = pd.DataFrame(data['comments'])
    else:
        df = pd.DataFrame(data)
    
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"Đã chuyển đổi {json_path} -> {parquet_path}")
    return parquet_path


def save_results_to_pickle(results, output_path):
    import pickle
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Đã lưu kết quả vào {output_path}")


def load_results_from_pickle(input_path):
    import pickle
    
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    
    return results
