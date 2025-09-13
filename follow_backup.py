import redis
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
import re
from datetime import datetime
from dataclasses import dataclass, asdict
import argparse
import numpy as np
import requests
import time
from pathlib import Path
import pickle
import os

# Try to import RAG dependencies with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGDocument:
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class FollowUpQuestion:
    question: str
    relevance_score: float
    source_docs: List[str]
    category: str
    priority: int

class MedicalRAGRetriever:
    """RAG retriever for medical knowledge and guidelines"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.documents = []
        self.embeddings = []
        self.embedding_model = None
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.use_embeddings = SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE
        
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                logger.info(f"Loaded embedding model: {embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Falling back to TF-IDF")
                self.use_embeddings = False
        
        if not self.use_embeddings and SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            logger.info("Using TF-IDF for document retrieval")
        
    def add_documents(self, documents: List[Dict]):
        """Add documents to the knowledge base"""
        for doc_data in documents:
            doc = RAGDocument(
                id=doc_data.get('id', f"doc_{len(self.documents)}"),
                content=doc_data.get('content', ''),
                metadata=doc_data.get('metadata', {})
            )
            self.documents.append(doc)
        
        self._build_index()
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def load_documents_from_file(self, file_path: str):
        """Load documents from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            self.add_documents(documents)
        except Exception as e:
            logger.error(f"Failed to load documents from {file_path}: {e}")
    
    def load_documents_from_redis(self, redis_client, key_pattern="transcription_stream"):
        """Load documents from Redis"""
        try:
            keys = redis_client.keys(key_pattern)
            documents = []
            for key in keys:
                doc_data = redis_client.hgetall(key)
                if doc_data:
                    documents.append({
                        'id': key.decode() if isinstance(key, bytes) else key,
                        'content': doc_data.get('content', ''),
                        'metadata': json.loads(doc_data.get('metadata', '{}'))
                    })
            
            if documents:
                self.add_documents(documents)
                logger.info(f"Loaded {len(documents)} documents from Redis")
            else:
                logger.warning("No documents found in Redis with pattern: " + key_pattern)
                
        except Exception as e:
            logger.error(f"Failed to load documents from Redis: {e}")
    
    def _build_index(self):
        """Build search index for documents"""
        if not self.documents:
            return
        
        contents = [doc.content for doc in self.documents]
        
        if self.use_embeddings:
            # Build FAISS index with embeddings
            embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
            self.embeddings = embeddings
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            
            logger.info(f"Built FAISS index with {len(embeddings)} embeddings")
            
        elif self.tfidf_vectorizer:
            # Build TF-IDF index
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
            logger.info(f"Built TF-IDF index with {len(contents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[RAGDocument, float]]:
        """Retrieve most relevant documents for a query"""
        if not self.documents:
            return []
        
        if self.use_embeddings and self.faiss_index:
            return self._retrieve_with_embeddings(query, top_k)
        elif self.tfidf_matrix is not None:
            return self._retrieve_with_tfidf(query, top_k)
        else:
            # Fallback to simple keyword matching
            return self._retrieve_with_keywords(query, top_k)
    
    def _retrieve_with_embeddings(self, query: str, top_k: int) -> List[Tuple[RAGDocument, float]]:
        """Retrieve using semantic embeddings"""
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def _retrieve_with_tfidf(self, query: str, top_k: int) -> List[Tuple[RAGDocument, float]]:
        """Retrieve using TF-IDF similarity"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def _retrieve_with_keywords(self, query: str, top_k: int) -> List[Tuple[RAGDocument, float]]:
        """Simple keyword-based retrieval as fallback"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            doc_words = set(doc.content.lower().split())
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / len(query_words) if query_words else 0
            
            if score > 0:
                results.append((doc, score))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class RAGFollowUpGenerator:
    """Generate follow-up questions using RAG"""
    
    def __init__(self, retriever: MedicalRAGRetriever):
        self.retriever = retriever
        self.redis_client = None
        
        # Question generation templates
        self.question_templates = {
            "symptom_detail": "Can you provide more details about {symptom}?",
            "timeline": "When did {condition} first start?",
            "severity": "How severe is your {symptom} on a scale of 1-10?",
            "triggers": "What triggers or worsens your {symptom}?",
            "medication": "Are you taking any medications for {condition}?",
            "impact": "How does {condition} affect your daily activities?",
            "family_history": "Is there a family history of {condition}?",
            "previous_treatment": "Have you received treatment for {condition} before?"
        }
    
    def connect_redis(self, host='localhost', port=6379, db=0):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def read_conversation_from_redis(self, stream_name='transcription_stream', count=100) -> str:
        """Read conversation from Redis stream"""
        try:
            messages = self.redis_client.xrange(stream_name, count=count)
            conversation_history = []
            
            for msg_id, fields in messages:
                text = fields.get('text', '')
                speaker = fields.get('speaker', 'unknown')
                timestamp = fields.get('timestamp', '')
                
                if text:
                    conversation_history.append(f"[{speaker}]: {text}")
            
            return "\n".join(conversation_history)
            
        except Exception as e:
            logger.error(f"Error reading from Redis: {e}")
            return ""
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from conversation text"""
        # Simple pattern-based entity extraction
        entities = {
            'symptoms': [],
            'conditions': [],
            'medications': [],
            'body_parts': [],
            'time_expressions': []
        }
        
        # Symptom patterns
        symptom_patterns = [
            r'\b(pain|ache|hurt|sore|burning|tingling|numbness)\b',
            r'\b(fever|chills|nausea|vomiting|dizziness|fatigue)\b',
            r'\b(cough|shortness of breath|chest pain|headache)\b',
            r'\b(swelling|rash|itching|bleeding)\b'
        ]
        
        # Condition patterns
        condition_patterns = [
            r'\b(diabetes|hypertension|asthma|arthritis|depression)\b',
            r'\b(heart disease|stroke|cancer|thyroid|kidney disease)\b',
            r'\b(high blood pressure|low blood sugar|migraine)\b'
        ]
        
        # Extract symptoms
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text.lower())
            entities['symptoms'].extend(matches)
        
        # Extract conditions
        for pattern in condition_patterns:
            matches = re.findall(pattern, text.lower())
            entities['conditions'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def generate_context_query(self, conversation: str, entities: Dict) -> str:
        """Generate search query from conversation context"""
        # Combine important entities and context
        query_parts = []
        
        if entities['conditions']:
            query_parts.append(f"medical conditions: {', '.join(entities['conditions'])}")
        
        if entities['symptoms']:
            query_parts.append(f"symptoms: {', '.join(entities['symptoms'])}")
        
        # Add recent conversation context
        recent_lines = conversation.split('\n')[-3:]  # Last 3 exchanges
        query_parts.extend(recent_lines)
        
        return " ".join(query_parts)
    
    def generate_followup_questions(self, conversation: str, max_questions: int = 3) -> List[FollowUpQuestion]:
        """Generate follow-up questions using RAG"""
        if not conversation.strip():
            return []
        
        # Extract medical entities from conversation
        entities = self.extract_medical_entities(conversation)
        
        # Generate search query
        search_query = self.generate_context_query(conversation, entities)
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(search_query, top_k=5)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found for query")
            return self._generate_fallback_questions(entities)
        
        # Generate questions based on retrieved knowledge
        questions = []
        
        for doc, relevance_score in retrieved_docs[:max_questions]:
            # Extract question templates from document metadata
            if 'questions' in doc.metadata:
                doc_questions = doc.metadata['questions']
                if isinstance(doc_questions, list):
                    for q in doc_questions[:2]:  # Max 2 questions per document
                        questions.append(FollowUpQuestion(
                            question=q,
                            relevance_score=relevance_score,
                            source_docs=[doc.id],
                            category=doc.metadata.get('category', 'general'),
                            priority=doc.metadata.get('priority', 2)
                        ))
            
            # Generate questions based on document content
            content_questions = self._extract_questions_from_content(
                doc.content, entities, relevance_score, doc.id
            )
            questions.extend(content_questions)
        
        # Remove duplicates and rank by relevance
        unique_questions = self._deduplicate_questions(questions)
        
        # Sort by priority and relevance
        unique_questions.sort(key=lambda q: (q.priority, -q.relevance_score))
        
        return unique_questions[:max_questions]
    
    def _extract_questions_from_content(self, content: str, entities: Dict, 
                                      relevance_score: float, doc_id: str) -> List[FollowUpQuestion]:
        """Extract or generate questions from document content"""
        questions = []
        
        # Look for explicit questions in content
        question_pattern = r'[A-Z][^.!?]*\?'
        found_questions = re.findall(question_pattern, content)
        
        for q in found_questions[:2]:  # Max 2 per document
            questions.append(FollowUpQuestion(
                question=q,
                relevance_score=relevance_score * 0.9,  # Slightly lower for extracted
                source_docs=[doc_id],
                category='extracted',
                priority=2
            ))
        
        # Generate template-based questions for entities
        for condition in entities.get('conditions', []):
            if 'medication' in content.lower():
                questions.append(FollowUpQuestion(
                    question=f"Are you taking any medications for {condition}?",
                    relevance_score=relevance_score * 0.8,
                    source_docs=[doc_id],
                    category='generated',
                    priority=1
                ))
        
        for symptom in entities.get('symptoms', []):
            if 'severity' in content.lower() or 'scale' in content.lower():
                questions.append(FollowUpQuestion(
                    question=f"On a scale of 1-10, how severe is your {symptom}?",
                    relevance_score=relevance_score * 0.8,
                    source_docs=[doc_id],
                    category='generated',
                    priority=2
                ))
        
        return questions
    
    def _deduplicate_questions(self, questions: List[FollowUpQuestion]) -> List[FollowUpQuestion]:
        """Remove duplicate questions"""
        seen_questions = set()
        unique_questions = []
        
        for q in questions:
            # Simple similarity check
            normalized_q = re.sub(r'\s+', ' ', q.question.lower().strip())
            if normalized_q not in seen_questions:
                seen_questions.add(normalized_q)
                unique_questions.append(q)
        
        return unique_questions
    
    def _generate_fallback_questions(self, entities: Dict) -> List[FollowUpQuestion]:
        """Generate basic questions when RAG retrieval fails"""
        fallback_questions = []
        
        # Generic medical questions
        generic_questions = [
            "Can you tell me more about your current symptoms?",
            "How long have you been experiencing these symptoms?",
            "Are you currently taking any medications?",
            "Have you seen a doctor about this before?",
            "How are these symptoms affecting your daily activities?"
        ]
        
        for i, q in enumerate(generic_questions[:3]):
            fallback_questions.append(FollowUpQuestion(
                question=q,
                relevance_score=0.5,
                source_docs=['fallback'],
                category='generic',
                priority=3
            ))
        
        return fallback_questions
    
    def save_session_context(self, session_id: str, context: Dict):
        """Save session context to Redis"""
        if self.redis_client:
            try:
                self.redis_client.hset(
                    f"session:{session_id}", 
                    mapping={
                        'context': json.dumps(context),
                        'timestamp': datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Failed to save session context: {e}")

def main():
    parser = argparse.ArgumentParser(description='Medical RAG Follow-up Question Generator')
    parser.add_argument('--docs-file', help='JSON file containing medical documents')
    parser.add_argument('--docs-redis-pattern', default='transcription_stream',
                       help='Redis key pattern for medical documents')
    parser.add_argument('--stream', default='transcription_stream',
                       help='Redis stream name for conversation')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--max-questions', type=int, default=3,
                       help='Maximum number of questions to generate')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        logger.info("Initializing Medical RAG system...")
        retriever = MedicalRAGRetriever(embedding_model_name=args.embedding_model)
        
        # Load documents
        if args.docs_file:
            retriever.load_documents_from_file(args.docs_file)
        
        # Load from Redis if available
        redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        retriever.load_documents_from_redis(redis_client, args.docs_redis_pattern)
        
        # Initialize question generator
        generator = RAGFollowUpGenerator(retriever)
        generator.connect_redis(host=args.redis_host, port=args.redis_port)
        
        # Read conversation from Redis
        print("=== Reading conversation from Redis ===")
        conversation = generator.read_conversation_from_redis(args.stream)
        
        if not conversation:
            print("No conversation data found in Redis stream.")
            return
        
        print("=== Current Conversation ===")
        print(conversation)
        print("=" * 50)
        
        # Generate follow-up questions
        print("\n=== Generating RAG-based Follow-up Questions ===")
        start_time = time.time()
        
        questions = generator.generate_followup_questions(
            conversation, 
            max_questions=args.max_questions
        )
        
        generation_time = time.time() - start_time
        
        print(f"\n=== Generated {len(questions)} Questions (in {generation_time:.2f}s) ===")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q.question}")
            print(f"   └─ Relevance: {q.relevance_score:.3f} | Category: {q.category} | Priority: {q.priority}")
            print(f"   └─ Sources: {', '.join(q.source_docs)}")
            print()
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()