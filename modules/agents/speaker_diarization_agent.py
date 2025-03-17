#!/usr/bin/env python3

import os
import torch
import numpy as np
import wave
import contextlib
from typing import Dict, List

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from modules.config import SAMPLE_RATE

class SpeakerDiarizationAgent:
    """
    Agent that handles speaker diarization.
    Identifies different speakers in an audio recording.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize embedding model
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )
        self.audio = Audio()
        self.user_embedding = None
        self.known_speakers = {}
        self.min_segment_duration = 1.0  
        self.similarity_threshold = 0.5 
        
    def capture_user_reference(self, audio_file: str) -> bool:
        """
        Capture a reference sample of the user's voice to identify them in future conversations.
        """
        try:
            with contextlib.closing(wave.open(audio_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            
            clip = Segment(0, duration)
            waveform, sample_rate = self.audio.crop(audio_file, clip)
            self.user_embedding = self.embedding_model(waveform[None])
            
            self.known_speakers["USER"] = self.user_embedding
            
            print("✓ User voice reference captured successfully")
            return True
            
        except Exception as e:
            print(f"Error capturing user reference: {e}")
            return False
    
    def _get_segment_embedding(self, audio_file: str, start: float, end: float) -> np.ndarray:
        clip = Segment(start, end)
        waveform, sample_rate = self.audio.crop(audio_file, clip)
        return self.embedding_model(waveform[None])
    
    def _compute_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0])
    
    def _manual_clustering(self, embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
        """
        Implement manual clustering based on cosine similarities
        for compatibility with older scikit-learn versions.
        """
        n_samples = embeddings.shape[0]
        
        if n_samples <= num_clusters:
            return np.arange(n_samples)
        
        normalized_embeddings = normalize(embeddings)
        
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        avg_similarities = np.mean(similarity_matrix, axis=1)
        cluster_centers = [np.argmin(avg_similarities)]
        
        while len(cluster_centers) < num_clusters:
            max_min_sim = -1
            next_center = -1
            
            for i in range(n_samples):
                if i in cluster_centers:
                    continue
                
                min_sim = min(similarity_matrix[i, center] for center in cluster_centers)
                
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    next_center = i
            
            if next_center != -1:
                cluster_centers.append(next_center)
        
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            similarities = [similarity_matrix[i, center] for center in cluster_centers]
            labels[i] = np.argmax(similarities)
        
        return labels
    
    def process_conversation(self, 
                            audio_file: str, 
                            segments: List[Dict], 
                            num_speakers: int = 2) -> List[Dict]:
        """
        Process an audio file containing multiple speakers and assign speaker labels.
        """
        try:
            print(f"Processing conversation with {len(segments)} segments, expecting {num_speakers} speakers")
            with contextlib.closing(wave.open(audio_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            
            valid_segments_indices = []
            valid_segments = []
            for i, segment in enumerate(segments):
                start = segment["start"]
                end = min(duration, segment["end"])
                if end - start >= self.min_segment_duration:
                    valid_segments.append(segment)
                    valid_segments_indices.append(i)
            
            print(f"Found {len(valid_segments)} valid segments for diarization")
            
            if len(valid_segments) < 2:
                print("Not enough valid segments, using all segments")
                valid_segments = segments
                valid_segments_indices = list(range(len(segments)))
            
            num_valid_segments = len(valid_segments)
            embeddings = np.zeros(shape=(num_valid_segments, 192))
            
            for i, segment in enumerate(valid_segments):
                start = segment["start"]
                end = min(duration, segment["end"])
                embeddings[i] = self._get_segment_embedding(audio_file, start, end)
            
            embeddings = np.nan_to_num(embeddings)
            
            effective_num_speakers = min(num_speakers, num_valid_segments)
            print(f"Clustering into {effective_num_speakers} speakers")
            
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=effective_num_speakers
                ).fit(embeddings)
                labels = clustering.labels_
                print("Using scikit-learn clustering")
            except Exception as e:
                print(f"Scikit-learn clustering failed: {e}")
                print("Falling back to manual clustering")
                labels = self._manual_clustering(embeddings, effective_num_speakers)
            
            user_cluster = None
            highest_similarity = -1.0
            
            if self.user_embedding is not None:
                cluster_similarities = {}
                
                for cluster_id in range(effective_num_speakers):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    
                    if len(cluster_indices) > 0:
                        similarities = []
                        for idx in cluster_indices:
                            sim = self._compute_embedding_similarity(
                                embeddings[idx], 
                                self.user_embedding
                            )
                            similarities.append(sim)
                        
                        max_similarity = max(similarities)
                        avg_similarity = np.mean(similarities)
                        cluster_similarities[cluster_id] = max_similarity
                        
                        print(f"Cluster {cluster_id} similarity to user: {max_similarity:.4f} (avg: {avg_similarity:.4f})")
                
                if cluster_similarities:
                    user_cluster, highest_similarity = max(
                        cluster_similarities.items(), 
                        key=lambda x: x[1]
                    )
                    
                    if highest_similarity < self.similarity_threshold:
                        print(f"No cluster matches user voice well enough. Best match: {highest_similarity:.4f}")
                        user_cluster = None
                    else:
                        print(f"User matched to cluster {user_cluster} with similarity {highest_similarity:.4f}")
            
            # Assign speaker labels to all segments based on clustering results
            for i, segment in enumerate(segments):
                if i not in valid_segments_indices:
                    seg_mid = (segment["start"] + segment["end"]) / 2
                    closest_valid_idx = min(
                        valid_segments_indices,
                        key=lambda idx: abs((segments[idx]["start"] + segments[idx]["end"]) / 2 - seg_mid)
                    )
                    cluster_id = labels[valid_segments_indices.index(closest_valid_idx)]
                else:
                    cluster_id = labels[valid_segments_indices.index(i)]
                
                if user_cluster is not None and cluster_id == user_cluster:
                    speaker_label = "USER"
                else:
                    other_id = cluster_id + 1
                    if user_cluster is not None and user_cluster < cluster_id:
                        other_id += 1
                    speaker_label = f"SPEAKER_{other_id}"
                
                segment["speaker"] = speaker_label
            
            return segments
            
        except Exception as e:
            print(f"Error in speaker diarization: {e}")
            import traceback
            traceback.print_exc()
            for segment in segments:
                segment["speaker"] = "UNKNOWN"
            return segments
    
    def identify_speaker(self, audio_file: str) -> str:
        """
        Identify the speaker in a short audio clip.
        Useful for determining if the current speaker is the user or someone else.
        """
        try:
            with contextlib.closing(wave.open(audio_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            
            embedding = self._get_segment_embedding(audio_file, 0, duration)
            
            if not self.known_speakers:
                return "UNKNOWN"
            
            similarities = {}
            for speaker_id, speaker_embedding in self.known_speakers.items():
                similarity = self._compute_embedding_similarity(embedding, speaker_embedding)
                similarities[speaker_id] = similarity
            
            best_match = max(similarities.items(), key=lambda x: x[1])
            speaker_id, similarity = best_match
            
            if similarity < self.similarity_threshold:
                print(f"Best speaker match ({speaker_id}) has too low similarity: {similarity:.4f}")
                return "UNKNOWN"
                
            print(f"Identified speaker as {speaker_id} with similarity {similarity:.4f}")
            return speaker_id
            
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return "UNKNOWN" 