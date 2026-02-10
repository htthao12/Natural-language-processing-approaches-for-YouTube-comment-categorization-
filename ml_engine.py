from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.sparse as sp


class MLTrainer:
    
    def __init__(self):
        self.algorithms = {
            'MultinomialNB': MultinomialNB(alpha=0.1),
            'LogisticRegressor': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            ),
            'RidgeClassifier': RidgeClassifier(
                alpha=1.0,
                random_state=42
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=50,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'SVM-Linear': LinearSVC(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'SVM-RBF': SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            )
        }
        
        self.label_transformer = LabelEncoder()
        self.champion_model = None
        self.champion_name = ""
        self.champion_score = 0
    
    def run_evaluation(self, X_train, X_test, y_train, y_test,
                      y_train_raw, y_test_raw, feature_type):
        
        print(f"\n{'='*70}")
        print(f"ĐÁNH GIÁ VỚI FEATURE: {feature_type}")
        print(f"{'='*70}")
        
        results_dict = {}
        best_cm = None
        best_local_score = 0
        best_local_name = None
        best_y_test = None
        best_y_pred = None
        
        for algo_name, classifier in self.algorithms.items():
            try:
                if algo_name == 'MultinomialNB':
                    if not sp.issparse(X_train) and np.any(X_train < 0):
                        classifier = GaussianNB()
                
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                f1_w = f1_score(y_test, y_pred, average='weighted')
                prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results_dict[algo_name] = {
                    'accuracy': acc,
                    'f1_score': f1_w,
                    'precision': prec_w,
                    'recall': rec_w,
                    'classifier': classifier
                }
                
                print(f"\n{algo_name}:")
                print(f"  Accuracy:  {acc:.4f}")
                print(f"  F1-Score:  {f1_w:.4f}")
                print(f"  Precision: {prec_w:.4f}")
                print(f"  Recall:    {rec_w:.4f}")
                print(classification_report(y_test, y_pred, zero_division=0))
                
                if acc > self.champion_score:
                    self.champion_score = acc
                    self.champion_model = classifier
                    self.champion_name = f"{feature_type} + {algo_name}"
                
                if acc > best_local_score:
                    best_local_score = acc
                    best_cm = confusion_matrix(y_test, y_pred)
                    best_local_name = algo_name
                    best_y_test = y_test
                    best_y_pred = y_pred
                    
            except Exception as e:
                print(f"\n{algo_name}: LỖI - {str(e)}")
                results_dict[algo_name] = {
                    'accuracy': 0,
                    'f1_score': 0,
                    'precision': 0,
                    'recall': 0,
                    'classifier': None
                }
        
        best_algo = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nModel tốt nhất cho {feature_type}: {best_algo[0]} (Accuracy: {best_algo[1]['accuracy']:.4f})")
        
        if best_cm is not None:
            print(f"\nConfusion Matrix cho model tốt nhất ({best_local_name}):")
            print(best_cm)
        
        return results_dict
    
    def get_champion_model(self):
        """Lấy model tốt nhất toàn cục"""
        return self.champion_model, self.champion_name, self.champion_score
    
    def predict(self, classifier, features):
        """Dự đoán với classifier"""
        return classifier.predict(features)
    
    def predict_proba(self, classifier, features):
        """Lấy xác suất dự đoán"""
        if hasattr(classifier, 'predict_proba'):
            return classifier.predict_proba(features)
        elif hasattr(classifier, 'decision_function'):
            return classifier.decision_function(features)
        return None


class PerformanceEvaluator:
    
    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Tính toán các metrics đánh giá"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def print_report(y_true, y_pred, title="Classification Report"):
        """In báo cáo đánh giá chi tiết"""
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        print(classification_report(y_true, y_pred, zero_division=0))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    
    @staticmethod
    def compare_models(results_dict):
        """So sánh hiệu suất các model"""
        print(f"\n{'='*70}")
        print("SO SÁNH HIỆU SUẤT CÁC MODEL")
        print(f"{'='*70}")
        
        # Sắp xếp theo accuracy
        sorted_results = sorted(
            results_dict.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f}")
        
        return sorted_results


class ExperimentLogger:
    
    def __init__(self):
        self.experiments = []
    
    def log_experiment(self, feature_name, model_name, metrics):
        """Ghi lại một thử nghiệm"""
        experiment = {
            'feature': feature_name,
            'model': model_name,
            'metrics': metrics
        }
        self.experiments.append(experiment)
    
    def get_best_experiment(self):
        """Lấy thử nghiệm tốt nhất"""
        if not self.experiments:
            return None
        
        best = max(
            self.experiments,
            key=lambda x: x['metrics']['accuracy']
        )
        return best
    
    def export_results(self, output_path):
        """Xuất kết quả ra file"""
        import pandas as pd
        
        data = []
        for exp in self.experiments:
            row = {
                'feature': exp['feature'],
                'model': exp['model'],
                **exp['metrics']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Đã xuất kết quả ra {output_path}")
    
    def print_summary(self):
        """In tóm tắt các thử nghiệm"""
        if not self.experiments:
            print("Chưa có thử nghiệm nào!")
            return
        
        print(f"\n{'='*70}")
        print("TÓM TẮT CÁC THỬ NGHIỆM")
        print(f"{'='*70}")
        
        for i, exp in enumerate(self.experiments, 1):
            print(f"\n[{i}] {exp['feature']} + {exp['model']}")
            print(f"    Accuracy: {exp['metrics']['accuracy']:.4f}")
            print(f"    F1-Score: {exp['metrics']['f1_score']:.4f}")
        
        best = self.get_best_experiment()
        if best:
            print(f"\n{'='*70}")
            print("THỬ NGHIỆM TỐT NHẤT")
            print(f"{'='*70}")
            print(f"Feature: {best['feature']}")
            print(f"Model: {best['model']}")
            print(f"Accuracy: {best['metrics']['accuracy']:.4f}")
            print(f"F1-Score: {best['metrics']['f1_score']:.4f}")
