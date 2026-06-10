import os
import csv
import argparse
import numpy as np

BASE_DIR = '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization'

def compute_scores(results_csv_path, target_env=None, w1=0.5, w2=0.3, w3=0.2):
    """
    Load results CSV, compute weighted scores for each model.
    If target_env given, weight mAP from that env more heavily.
    Returns sorted list of dicts: [{model, score, mAP50, robustness, fps_norm}]
    """
    if not os.path.exists(results_csv_path):
        print(f"Error: Results CSV not found: {results_csv_path}")
        return []
        
    model_data = {}
    max_fps = 0.0
    
    with open(results_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row['model']
            env = row['test_env']
            map50 = float(row['mAP50'])
            fps = float(row['fps'])
            
            if fps > max_fps:
                max_fps = fps
                
            if m not in model_data:
                model_data[m] = {'maps': {}, 'fps_list': []}
                
            model_data[m]['maps'][env] = map50
            model_data[m]['fps_list'].append(fps)
            
    scores = []
    for m, data in model_data.items():
        maps = list(data['maps'].values())
        avg_fps = np.mean(data['fps_list'])
        fps_norm = avg_fps / max_fps if max_fps > 0 else 0.0
        
        shift_variance = np.std(maps)
        robustness = max(0.0, 1.0 - shift_variance)
        
        if target_env and target_env in data['maps']:
            primary_map = data['maps'][target_env]
        else:
            primary_map = np.mean(maps)
            
        score = (w1 * primary_map) + (w2 * robustness) + (w3 * fps_norm)
        
        scores.append({
            'model': m,
            'score': score,
            'mAP50': primary_map,
            'robustness': robustness,
            'fps_norm': fps_norm,
            'avg_fps': avg_fps
        })
        
    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores

def print_recommendation(scores):
    """
    Print ranked model recommendation
    """
    if not scores:
        print("No scores computed.")
        return
        
    print("\nDEPLOYMENT RECOMMENDATION")
    print("═══════════════════════════════")
    
    for i, s in enumerate(scores):
        if i == 0:
            print(f"Rank {i+1}: {s['model']:<12} (score: {s['score']:.3f})")
            print(f"        mAP50: {s['mAP50']:.3f} | Robustness: {s['robustness']:.3f} | FPS: {s['avg_fps']:.1f}")
        else:
            print(f"Rank {i+1}: {s['model']:<12} (score: {s['score']:.3f})")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=f'{BASE_DIR}/benchmark/results/matrix_results.csv')
    parser.add_argument('--env', default=None, help='Target deployment environment')
    parser.add_argument('--w1', type=float, default=0.5)
    parser.add_argument('--w2', type=float, default=0.3)
    parser.add_argument('--w3', type=float, default=0.2)
    args = parser.parse_args()
    
    scores = compute_scores(args.results, args.env, args.w1, args.w2, args.w3)
    print_recommendation(scores)
