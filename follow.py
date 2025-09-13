import redis
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
import re
from datetime import datetime, timedelta
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
class ConversationLine:
    """Represents a single line of conversation"""
    timestamp: str
    speaker: str
    text: str
    line_id: Optional[str] = None

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
    context_entities: List[str] = None

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
        self.conversation_cache = {}
        
        # Enhanced question generation templates
        self.question_templates = {
            "symptom_detail": [
                "Can you describe your {entity} in more detail?",
                "Where exactly do you feel the {entity}?",
                "What does the {entity} feel like exactly?"
            ],
            "timeline": [
                "When did the {entity} first start?",
                "How long have you been experiencing {entity}?",
                "Has the {entity} gotten better or worse over time?"
            ],
            "severity": [
                "How severe is your {entity} on a scale of 1-10?",
                "How would you rate the intensity of your {entity}?",
                "Is the {entity} mild, moderate, or severe?"
            ],
            "triggers": [
                "What seems to trigger or worsen your {entity}?",
                "Have you noticed anything that makes the {entity} better or worse?",
                "Does anything specific bring on your {entity}?"
            ],
            "medication": [
                "Are you currently taking any medications for {entity}?",
                "What treatments have you tried for {entity}?",
                "Have you taken any over-the-counter medications for {entity}?"
            ],
            "impact": [
                "How does the {entity} affect your daily activities?",
                "Is the {entity} interfering with your sleep or work?",
                "How much is the {entity} limiting what you can do?"
            ],
            "associated_symptoms": [
                "Are you experiencing any other symptoms along with {entity}?",
                "What other symptoms have you noticed besides {entity}?",
                "Have you had any additional symptoms since the {entity} started?"
            ],
            "frequency": [
                "How often do you experience {entity}?",
                "Is the {entity} constant or does it come and go?",
                "How frequently does the {entity} occur?"
            ]
        }
        
        # Medical entity patterns (enhanced)
        self.medical_patterns = {
            'symptoms': [
                r'\b(pain|ache|hurt|sore|burning|tingling|numbness|stiffness)\b',
                r'\b(fever|chills|nausea|vomiting|dizziness|fatigue|weakness)\b',
                r'\b(cough|shortness of breath|chest pain|headache|migraine)\b',
                r'\b(swelling|rash|itching|bleeding|bruising|discharge)\b',
                r'\b(diarrhea|constipation|bloating|heartburn|indigestion)\b',
                r'\b(insomnia|sleep problems|anxiety|depression|mood changes)\b'
            ],
            'conditions': [
                r'\b(diabetes|hypertension|asthma|arthritis|depression|anxiety)\b',
                r'\b(heart disease|stroke|cancer|thyroid|kidney disease)\b',
                r'\b(high blood pressure|low blood sugar|migraine|epilepsy)\b',
                r'\b(copd|pneumonia|bronchitis|allergies|eczema|psoriasis)\b'
            ],
            'medications': [
                r'\b(aspirin|ibuprofen|acetaminophen|tylenol|advil|motrin)\b',
                r'\b(insulin|metformin|lisinopril|atorvastatin|omeprazole)\b',
                r'\b(antibiotics|steroids|painkillers|blood thinners)\b'
            ],
            'body_parts': [
                r'\b(head|neck|shoulder|arm|hand|chest|back|stomach|leg|foot)\b',
                r'\b(heart|lungs|liver|kidney|brain|spine|joint|muscle)\b'
            ],
            'time_expressions': [
                r'\b(yesterday|today|last week|last month|this morning|tonight)\b',
                r'\b(\d+\s+(days?|weeks?|months?|years?)\s+ago)\b',
                r'\b(since|for|about|around|approximately)\s+\d+\b'
            ]
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
    
    def read_conversation_lines(self, conversation_key='medical_conversation', 
                              max_lines=50, recent_minutes=30) -> List[ConversationLine]:
        """Read conversation lines from Redis with time filtering"""
        conversation_lines = []
        
        try:
            # Try multiple storage patterns
            storage_patterns = [
                # Hash storage: conversation_key with line numbers as fields
                ('hash', conversation_key),
                # List storage: conversation_key as a list
                ('list', conversation_key),
                # Stream storage: conversation_key as a stream
                ('stream', conversation_key),
                # Individual keys: conversation_key:line_number
                ('keys', f"{conversation_key}:*")
            ]
            
            for storage_type, pattern in storage_patterns:
                if storage_type == 'hash':
                    lines = self._read_from_hash(pattern)
                elif storage_type == 'list':
                    lines = self._read_from_list(pattern)
                elif storage_type == 'stream':
                    lines = self._read_from_stream(pattern)
                elif storage_type == 'keys':
                    lines = self._read_from_individual_keys(pattern)
                
                if lines:
                    conversation_lines = lines
                    logger.info(f"Read {len(lines)} conversation lines from {storage_type} storage")
                    break
            
            if not conversation_lines:
                logger.warning("No conversation data found in any storage pattern")
                return []
            
            # Filter by time if recent_minutes is specified
            if recent_minutes:
                conversation_lines = self._filter_recent_lines(conversation_lines, recent_minutes)
            
            # Limit number of lines
            conversation_lines = conversation_lines[-max_lines:] if max_lines else conversation_lines
            
            return conversation_lines
            
        except Exception as e:
            logger.error(f"Error reading conversation from Redis: {e}")
            return []
    
    def _read_from_hash(self, key) -> List[ConversationLine]:
        """Read from hash storage pattern"""
        try:
            if not self.redis_client.exists(key):
                return []
            
            hash_data = self.redis_client.hgetall(key)
            lines = []
            
            # Sort by field name (assuming numeric line numbers)
            sorted_fields = sorted(hash_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            
            for field in sorted_fields:
                line_data = json.loads(hash_data[field]) if hash_data[field].startswith('{') else hash_data[field]
                
                if isinstance(line_data, dict):
                    lines.append(ConversationLine(
                        timestamp=line_data.get('timestamp', ''),
                        speaker=line_data.get('speaker', 'unknown'),
                        text=line_data.get('text', ''),
                        line_id=field
                    ))
                else:
                    # Simple text format
                    lines.append(ConversationLine(
                        timestamp=datetime.now().isoformat(),
                        speaker='unknown',
                        text=str(line_data),
                        line_id=field
                    ))
            
            return lines
        except Exception as e:
            logger.error(f"Error reading from hash {key}: {e}")
            return []
    
    def _read_from_list(self, key) -> List[ConversationLine]:
        """Read from list storage pattern"""
        try:
            if not self.redis_client.exists(key):
                return []
            
            list_data = self.redis_client.lrange(key, 0, -1)
            lines = []
            
            for i, item in enumerate(list_data):
                try:
                    line_data = json.loads(item)
                    lines.append(ConversationLine(
                        timestamp=line_data.get('timestamp', ''),
                        speaker=line_data.get('speaker', 'unknown'),
                        text=line_data.get('text', ''),
                        line_id=str(i)
                    ))
                except json.JSONDecodeError:
                    # Simple text format
                    lines.append(ConversationLine(
                        timestamp=datetime.now().isoformat(),
                        speaker='unknown',
                        text=item,
                        line_id=str(i)
                    ))
            
            return lines
        except Exception as e:
            logger.error(f"Error reading from list {key}: {e}")
            return []
    
    def _read_from_stream(self, key) -> List[ConversationLine]:
        """Read from stream storage pattern"""
        try:
            if not self.redis_client.exists(key):
                return []
            
            messages = self.redis_client.xrange(key)
            lines = []
            
            for msg_id, fields in messages:
                lines.append(ConversationLine(
                    timestamp=fields.get('timestamp', ''),
                    speaker=fields.get('speaker', 'unknown'),
                    text=fields.get('text', ''),
                    line_id=msg_id
                ))
            
            return lines
        except Exception as e:
            logger.error(f"Error reading from stream {key}: {e}")
            return []
    
    def _read_from_individual_keys(self, pattern) -> List[ConversationLine]:
        """Read from individual key storage pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if not keys:
                return []
            
            # Sort keys by line number if possible
            def extract_line_number(key):
                try:
                    return int(key.split(':')[-1])
                except:
                    return 0
            
            sorted_keys = sorted(keys, key=extract_line_number)
            lines = []
            
            for key in sorted_keys:
                data = self.redis_client.get(key)
                if data:
                    try:
                        line_data = json.loads(data)
                        lines.append(ConversationLine(
                            timestamp=line_data.get('timestamp', ''),
                            speaker=line_data.get('speaker', 'unknown'),
                            text=line_data.get('text', ''),
                            line_id=key
                        ))
                    except json.JSONDecodeError:
                        lines.append(ConversationLine(
                            timestamp=datetime.now().isoformat(),
                            speaker='unknown',
                            text=data,
                            line_id=key
                        ))
            
            return lines
        except Exception as e:
            logger.error(f"Error reading from individual keys {pattern}: {e}")
            return []
    
    def _filter_recent_lines(self, lines: List[ConversationLine], minutes: int) -> List[ConversationLine]:
        """Filter conversation lines to only recent ones"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            filtered_lines = []
            
            for line in lines:
                if line.timestamp:
                    try:
                        line_time = datetime.fromisoformat(line.timestamp.replace('Z', '+00:00'))
                        if line_time >= cutoff_time:
                            filtered_lines.append(line)
                    except:
                        # If timestamp parsing fails, include the line
                        filtered_lines.append(line)
                else:
                    # If no timestamp, include the line
                    filtered_lines.append(line)
            
            return filtered_lines
        except Exception as e:
            logger.error(f"Error filtering recent lines: {e}")
            return lines  # Return all lines if filtering fails
    
    def extract_medical_entities(self, conversation_lines: List[ConversationLine]) -> Dict[str, List[str]]:
        """Extract medical entities from conversation lines"""
        entities = {
            'symptoms': set(),
            'conditions': set(),
            'medications': set(),
            'body_parts': set(),
            'time_expressions': set()
        }
        
        # Combine all conversation text
        full_text = ' '.join([line.text for line in conversation_lines])
        
        # Extract entities using patterns
        for entity_type, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text.lower(), re.IGNORECASE)
                entities[entity_type].update(matches)
        
        # Convert sets to lists and remove empty strings
        for key in entities:
            entities[key] = [item for item in list(entities[key]) if item.strip()]
        
        return entities
    
    def analyze_conversation_context(self, conversation_lines: List[ConversationLine]) -> Dict:
        """Analyze conversation context for better question generation"""
        context = {
            'total_lines': len(conversation_lines),
            'speakers': set(),
            'recent_topics': [],
            'question_asked': False,
            'incomplete_responses': [],
            'entities': {},
            'conversation_flow': []
        }
        
        # Analyze each line
        for i, line in enumerate(conversation_lines):
            context['speakers'].add(line.speaker)
            
            # Check for questions
            if '?' in line.text:
                context['question_asked'] = True
            
            # Check for incomplete responses (very short responses)
            if len(line.text.split()) < 3 and line.speaker != 'system':
                context['incomplete_responses'].append({
                    'line_id': line.line_id,
                    'text': line.text,
                    'speaker': line.speaker
                })
            
            # Build conversation flow
            context['conversation_flow'].append({
                'speaker': line.speaker,
                'text_length': len(line.text),
                'has_question': '?' in line.text,
                'timestamp': line.timestamp
            })
        
        # Extract entities
        context['entities'] = self.extract_medical_entities(conversation_lines)
        
        # Identify recent topics (last few lines)
        recent_lines = conversation_lines[-3:] if len(conversation_lines) >= 3 else conversation_lines
        context['recent_topics'] = [line.text for line in recent_lines]
        
        return context
    
    def generate_context_aware_query(self, conversation_lines: List[ConversationLine], 
                                   context: Dict) -> str:
        """Generate search query based on conversation context"""
        query_components = []
        
        # Add recent high-priority entities
        entities = context['entities']
        if entities['conditions']:
            query_components.append(f"conditions: {' '.join(list(entities['conditions'])[:3])}")
        
        if entities['symptoms']:
            query_components.append(f"symptoms: {' '.join(list(entities['symptoms'])[:3])}")
        
        # Add recent conversation context
        recent_text = ' '.join(context['recent_topics'])
        if recent_text:
            query_components.append(f"context: {recent_text[:200]}")  # Limit length
        
        # Add speaker context if medical professional is involved
        if 'doctor' in context['speakers'] or 'physician' in context['speakers']:
            query_components.append("medical consultation")
        
        return ' '.join(query_components)
    
    def generate_followup_questions(self, conversation_key='medical_conversation', 
                                  max_questions: int = 3, max_lines: int = 50,
                                  recent_minutes: int = 30) -> List[FollowUpQuestion]:
        """Generate context-aware follow-up questions"""
        
        # Read conversation lines
        conversation_lines = self.read_conversation_lines(
            conversation_key, max_lines, recent_minutes
        )
        
        if not conversation_lines:
            logger.warning("No conversation data found")
            return self._generate_fallback_questions({})
        
        # Analyze conversation context
        context = self.analyze_conversation_context(conversation_lines)
        
        # Generate search query
        search_query = self.generate_context_aware_query(conversation_lines, context)
        
        if not search_query.strip():
            logger.warning("Empty search query generated")
            return self._generate_fallback_questions(context['entities'])
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(search_query, top_k=5)
        
        if not retrieved_docs:
            logger.warning(f"No relevant documents found for query: {search_query}")
            return self._generate_fallback_questions(context['entities'])
        
        # Generate questions
        questions = []
        
        # RAG-based questions from retrieved documents
        for doc, relevance_score in retrieved_docs:
            doc_questions = self._extract_questions_from_document(
                doc, context, relevance_score
            )
            questions.extend(doc_questions)
        
        # Template-based questions for identified entities
        template_questions = self._generate_template_questions(
            context['entities'], context
        )
        questions.extend(template_questions)
        
        # Context-specific questions based on conversation flow
        flow_questions = self._generate_flow_based_questions(context)
        questions.extend(flow_questions)
        
        # Deduplicate and rank questions
        unique_questions = self._deduplicate_and_rank_questions(questions, context)
        
        return unique_questions[:max_questions]
    
    def _extract_questions_from_document(self, doc: RAGDocument, context: Dict, 
                                       relevance_score: float) -> List[FollowUpQuestion]:
        """Extract questions from a retrieved document"""
        questions = []
        
        # Look for explicit questions in document
        question_patterns = [
            r'[A-Z][^.!?]*\?',  # Standard questions
            r'Ask about [^.!?]*[.?]',  # Clinical prompts
            r'Consider asking [^.!?]*[.?]',  # Assessment prompts
        ]
        
        for pattern in question_patterns:
            found_questions = re.findall(pattern, doc.content)
            for q in found_questions[:2]:  # Limit per pattern
                if len(q.split()) >= 4:  # Filter out very short questions
                    questions.append(FollowUpQuestion(
                        question=q.strip(),
                        relevance_score=relevance_score * 0.9,
                        source_docs=[doc.id],
                        category='document_extracted',
                        priority=1,
                        context_entities=list(context['entities']['symptoms'])[:3]
                    ))
        
        # Generate questions based on document metadata
        if 'follow_up_questions' in doc.metadata:
            metadata_questions = doc.metadata['follow_up_questions']
            if isinstance(metadata_questions, list):
                for q in metadata_questions[:2]:
                    questions.append(FollowUpQuestion(
                        question=q,
                        relevance_score=relevance_score,
                        source_docs=[doc.id],
                        category='metadata',
                        priority=1,
                        context_entities=list(context['entities']['conditions'])[:3]
                    ))
        
        return questions
    
    def _generate_template_questions(self, entities: Dict, context: Dict) -> List[FollowUpQuestion]:
        """Generate questions using templates and extracted entities"""
        questions = []
        
        # Prioritize entities based on context
        priority_entities = {
            'symptoms': list(entities.get('symptoms', []))[:2],
            'conditions': list(entities.get('conditions', []))[:2],
            'medications': list(entities.get('medications', []))[:1]
        }
        
        for entity_type, entity_list in priority_entities.items():
            for entity in entity_list:
                # Choose appropriate question templates
                suitable_templates = self._choose_templates_for_entity(
                    entity, entity_type, context
                )
                
                for template_category, templates in suitable_templates.items():
                    if templates:
                        question = templates[0].format(entity=entity)
                        questions.append(FollowUpQuestion(
                            question=question,
                            relevance_score=0.8,
                            source_docs=['template'],
                            category=f'template_{template_category}',
                            priority=2,
                            context_entities=[entity]
                        ))
        
        return questions
    
    def _choose_templates_for_entity(self, entity: str, entity_type: str, context: Dict) -> Dict[str, List[str]]:
        """Choose appropriate question templates based on entity and context"""
        suitable_templates = {}
        
        # Check what hasn't been asked yet based on conversation flow
        asked_about_severity = any('scale' in topic.lower() or 'severe' in topic.lower() 
                                 for topic in context.get('recent_topics', []))
        asked_about_timeline = any('when' in topic.lower() or 'started' in topic.lower() 
                                 for topic in context.get('recent_topics', []))
        asked_about_medication = any('medication' in topic.lower() or 'taking' in topic.lower() 
                                   for topic in context.get('recent_topics', []))
        
        # Select templates based on what hasn't been covered
        if not asked_about_severity and entity_type == 'symptoms':
            suitable_templates['severity'] = self.question_templates['severity']
        
        if not asked_about_timeline:
            suitable_templates['timeline'] = self.question_templates['timeline']
        
        if not asked_about_medication and entity_type in ['conditions', 'symptoms']:
            suitable_templates['medication'] = self.question_templates['medication']
        
        # Always include detail questions for symptoms
        if entity_type == 'symptoms':
            suitable_templates['symptom_detail'] = self.question_templates['symptom_detail']
        
        # Add trigger questions for chronic conditions
        chronic_conditions = ['arthritis', 'migraine', 'asthma', 'depression', 'anxiety']
        if entity_type == 'conditions' and entity.lower() in chronic_conditions:
            suitable_templates['triggers'] = self.question_templates['triggers']
        
        return suitable_templates
    
    def _generate_flow_based_questions(self, context: Dict) -> List[FollowUpQuestion]:
        """Generate questions based on conversation flow analysis"""
        questions = []
        
        # Check for incomplete responses
        if context['incomplete_responses']:
            questions.append(FollowUpQuestion(
                question="Could you provide more details about what you mentioned?",
                relevance_score=0.9,
                source_docs=['flow_analysis'],
                category='clarification',
                priority=1,
                context_entities=[]
            ))
        
        # Check if no questions have been asked yet
        if not context['question_asked'] and context['total_lines'] > 2:
            questions.append(FollowUpQuestion(
                question="What brings you in today? What's your main concern?",
                relevance_score=0.95,
                source_docs=['flow_analysis'],
                category='opening',
                priority=1,
                context_entities=[]
            ))
        
        # Check for follow-up needed based on conversation length
        if context['total_lines'] >= 5:
            recent_speaker_changes = self._analyze_speaker_pattern(context['conversation_flow'])
            if recent_speaker_changes < 2:  # One-sided conversation
                questions.append(FollowUpQuestion(
                    question="How are you feeling about what we've discussed so far?",
                    relevance_score=0.7,
                    source_docs=['flow_analysis'],
                    category='engagement',
                    priority=3,
                    context_entities=[]
                ))
        
        return questions
    
    def _analyze_speaker_pattern(self, conversation_flow: List[Dict]) -> int:
        """Analyze speaker change patterns in recent conversation"""
        recent_flow = conversation_flow[-5:] if len(conversation_flow) >= 5 else conversation_flow
        speaker_changes = 0
        
        for i in range(1, len(recent_flow)):
            if recent_flow[i]['speaker'] != recent_flow[i-1]['speaker']:
                speaker_changes += 1
        
        return speaker_changes
    
    def _deduplicate_and_rank_questions(self, questions: List[FollowUpQuestion], 
                                      context: Dict) -> List[FollowUpQuestion]:
        """Remove duplicates and rank questions by relevance and priority"""
        if not questions:
            return []
        
        # Remove duplicates based on question similarity
        unique_questions = []
        seen_questions = set()
        
        for q in questions:
            # Normalize question for comparison
            normalized = self._normalize_question(q.question)
            
            if normalized not in seen_questions:
                seen_questions.add(normalized)
                unique_questions.append(q)
        
        # Rank questions
        ranked_questions = self._rank_questions(unique_questions, context)
        
        return ranked_questions
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for duplicate detection"""
        # Remove punctuation, extra whitespace, and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', question.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common question words to focus on content
        stop_words = {'what', 'how', 'when', 'where', 'why', 'can', 'could', 'would', 'do', 'does', 'did', 'are', 'is', 'have', 'has'}
        words = [w for w in normalized.split() if w not in stop_words]
        
        return ' '.join(words)
    
    def _rank_questions(self, questions: List[FollowUpQuestion], context: Dict) -> List[FollowUpQuestion]:
        """Rank questions based on multiple factors"""
        
        def calculate_score(q: FollowUpQuestion) -> float:
            score = q.relevance_score
            
            # Priority boost (lower priority number = higher boost)
            priority_boost = (4 - q.priority) * 0.2
            score += priority_boost
            
            # Context relevance boost
            if q.context_entities:
                entity_overlap = len(set(q.context_entities) & 
                                   set(context['entities'].get('symptoms', []) + 
                                       context['entities'].get('conditions', [])))
                score += entity_overlap * 0.1
            
            # Category-based adjustments
            category_weights = {
                'opening': 0.3,
                'clarification': 0.25,
                'document_extracted': 0.2,
                'template_severity': 0.15,
                'template_timeline': 0.15,
                'metadata': 0.1,
                'engagement': 0.05
            }
            
            category_weight = category_weights.get(q.category, 0)
            score += category_weight
            
            return score
        
        # Calculate scores and sort
        for q in questions:
            q.relevance_score = calculate_score(q)
        
        return sorted(questions, key=lambda x: x.relevance_score, reverse=True)
    
    def _generate_fallback_questions(self, entities: Dict) -> List[FollowUpQuestion]:
        """Generate basic questions when RAG retrieval fails"""
        fallback_questions = []
        
        # Entity-specific fallback questions
        if entities.get('symptoms'):
            symptom = list(entities['symptoms'])[0]
            fallback_questions.append(FollowUpQuestion(
                question=f"Can you tell me more about your {symptom}?",
                relevance_score=0.6,
                source_docs=['fallback'],
                category='fallback_symptom',
                priority=2,
                context_entities=[symptom]
            ))
        
        if entities.get('conditions'):
            condition = list(entities['conditions'])[0]
            fallback_questions.append(FollowUpQuestion(
                question=f"How long have you been dealing with {condition}?",
                relevance_score=0.6,
                source_docs=['fallback'],
                category='fallback_condition',
                priority=2,
                context_entities=[condition]
            ))
        
        # Generic fallback questions
        generic_questions = [
            "What's your main concern today?",
            "Can you describe your symptoms in more detail?",
            "When did you first notice these symptoms?",
            "How are these symptoms affecting your daily life?",
            "Are you currently taking any medications?"
        ]
        
        for i, q in enumerate(generic_questions[:3]):
            fallback_questions.append(FollowUpQuestion(
                question=q,
                relevance_score=0.5,
                source_docs=['fallback'],
                category='generic',
                priority=3,
                context_entities=[]
            ))
        
        return fallback_questions
    
    def save_session_context(self, session_id: str, context: Dict, questions: List[FollowUpQuestion]):
        """Save session context and generated questions to Redis"""
        if self.redis_client:
            try:
                session_data = {
                    'context': json.dumps(context, default=str),
                    'questions': json.dumps([asdict(q) for q in questions], default=str),
                    'timestamp': datetime.now().isoformat(),
                    'question_count': len(questions)
                }
                
                self.redis_client.hset(f"session:{session_id}", mapping=session_data)
                
                # Set expiration (24 hours)
                self.redis_client.expire(f"session:{session_id}", 86400)
                
                logger.info(f"Saved session context for {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to save session context: {e}")
    
    def get_conversation_summary(self, conversation_key='medical_conversation') -> Dict:
        """Get a summary of the current conversation"""
        conversation_lines = self.read_conversation_lines(conversation_key)
        
        if not conversation_lines:
            return {'error': 'No conversation data found'}
        
        context = self.analyze_conversation_context(conversation_lines)
        
        summary = {
            'total_lines': context['total_lines'],
            'speakers': list(context['speakers']),
            'entities_found': {
                key: len(value) for key, value in context['entities'].items()
            },
            'recent_topics': context['recent_topics'][-3:],  # Last 3 topics
            'has_incomplete_responses': len(context['incomplete_responses']) > 0,
            'conversation_active': context['total_lines'] > 0
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Enhanced Medical RAG Follow-up Question Generator')
    parser.add_argument('--docs-file', help='JSON file containing medical documents')
    parser.add_argument('--docs-redis-pattern', default='transcription_stream', 
                       help='Redis key pattern for medical documents')
    parser.add_argument('--conversation-key', default='medical_conversation',
                       help='Redis key for conversation data')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--max-questions', type=int, default=3,
                       help='Maximum number of questions to generate')
    parser.add_argument('--max-lines', type=int, default=50,
                       help='Maximum conversation lines to analyze')
    parser.add_argument('--recent-minutes', type=int, default=30,
                       help='Only analyze conversation from last N minutes')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--session-id', help='Session ID for context saving')
    parser.add_argument('--summary', action='store_true',
                       help='Show conversation summary')
    parser.add_argument('--watch', action='store_true',
                       help='Watch for conversation updates')
    parser.add_argument('--watch-interval', type=int, default=10,
                       help='Watch interval in seconds')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        logger.info("Initializing Enhanced Medical RAG system...")
        retriever = MedicalRAGRetriever(embedding_model_name=args.embedding_model)
        
        # Load documents
        if args.docs_file and os.path.exists(args.docs_file):
            retriever.load_documents_from_file(args.docs_file)
            logger.info(f"Loaded documents from file: {args.docs_file}")
        
        # Load from Redis if available
        try:
            redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
            redis_client.ping()
            retriever.load_documents_from_redis(redis_client, args.docs_redis_pattern)
        except Exception as e:
            logger.warning(f"Could not connect to Redis for document loading: {e}")
        
        # Initialize question generator
        generator = RAGFollowUpGenerator(retriever)
        generator.connect_redis(host=args.redis_host, port=args.redis_port)
        
        if args.summary:
            # Show conversation summary
            print("=== Conversation Summary ===")
            summary = generator.get_conversation_summary(args.conversation_key)
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("=" * 40)
        
        if args.watch:
            # Watch mode - continuously monitor conversation
            print(f"Watching conversation '{args.conversation_key}' for updates...")
            print(f"Update interval: {args.watch_interval} seconds")
            print("Press Ctrl+C to stop watching.\n")
            
            last_line_count = 0
            
            while True:
                try:
                    # Check for new conversation data
                    conversation_lines = generator.read_conversation_lines(
                        args.conversation_key, args.max_lines, args.recent_minutes
                    )
                    
                    if len(conversation_lines) > last_line_count:
                        print(f"\n=== New Activity Detected ({datetime.now().strftime('%H:%M:%S')}) ===")
                        
                        # Show recent conversation
                        recent_lines = conversation_lines[last_line_count:]
                        for line in recent_lines:
                            print(f"[{line.speaker}]: {line.text}")
                        
                        # Generate questions
                        questions = generator.generate_followup_questions(
                            args.conversation_key,
                            max_questions=args.max_questions,
                            max_lines=args.max_lines,
                            recent_minutes=args.recent_minutes
                        )
                        
                        print(f"\n=== Suggested Follow-up Questions ===")
                        for i, q in enumerate(questions, 1):
                            print(f"{i}. {q.question}")
                            print(f"   └─ Score: {q.relevance_score:.3f} | Category: {q.category}")
                            if q.context_entities:
                                print(f"   └─ Context: {', '.join(q.context_entities)}")
                        print("=" * 50)
                        
                        last_line_count = len(conversation_lines)
                        
                        # Save session context if session ID provided
                        if args.session_id:
                            context = generator.analyze_conversation_context(conversation_lines)
                            generator.save_session_context(args.session_id, context, questions)
                    
                    time.sleep(args.watch_interval)
                    
                except KeyboardInterrupt:
                    print("\nStopping watch mode...")
                    break
        else:
            # Single run mode
            print("=== Reading Current Conversation ===")
            conversation_lines = generator.read_conversation_lines(
                args.conversation_key, args.max_lines, args.recent_minutes
            )
            
            if not conversation_lines:
                print("No conversation data found.")
                return
            
            print(f"Found {len(conversation_lines)} conversation lines")
            for line in conversation_lines[-5:]:  # Show last 5 lines
                print(f"[{line.speaker}]: {line.text[:100]}{'...' if len(line.text) > 100 else ''}")
            print("=" * 60)
            
            # Generate follow-up questions
            print("\n=== Generating Context-Aware Follow-up Questions ===")
            start_time = time.time()
            
            questions = generator.generate_followup_questions(
                args.conversation_key,
                max_questions=args.max_questions,
                max_lines=args.max_lines,
                recent_minutes=args.recent_minutes
            )
            
            generation_time = time.time() - start_time
            
            print(f"\n=== Generated {len(questions)} Questions (in {generation_time:.2f}s) ===")
            for i, q in enumerate(questions, 1):
                print(f"\n{i}. {q.question}")
                print(f"   ├─ Relevance Score: {q.relevance_score:.3f}")
                print(f"   ├─ Category: {q.category}")
                print(f"   ├─ Priority: {q.priority}")
                if q.context_entities:
                    print(f"   ├─ Context Entities: {', '.join(q.context_entities)}")
                print(f"   └─ Sources: {', '.join(q.source_docs)}")
            
            # Save session context if session ID provided
            if args.session_id:
                context = generator.analyze_conversation_context(conversation_lines)
                generator.save_session_context(args.session_id, context, questions)
                print(f"\nSession context saved for: {args.session_id}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()