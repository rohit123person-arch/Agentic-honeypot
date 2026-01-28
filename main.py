"""
Agentic Honeypot API - CPU-Optimized for Render Free Tier
Lightweight LLM integration using distilgpt2 for fast CPU inference

Optimized for:
- Low memory usage (< 512MB)
- Fast response times on CPU
- Render free tier constraints
"""

from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
try:
    from pydantic import ConfigDict
    PYDANTIC_V2 = True
except ImportError:
    PYDANTIC_V2 = False
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import time
import re
import uuid
import os
import logging
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Hugging Face / Transformers imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: transformers not installed. Using fallback responses.")

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURATION
# =========================
API_KEY = os.getenv("HONEYPOT_API_KEY", "hp_9f8a7b6c5d4e3f2a")
AUTH_DISABLED = os.getenv("HONEYPOT_DEV_DISABLE_AUTH") == "1"
MAX_TURNS = 8
SCAM_KEYWORD_THRESHOLD = 2

# CPU-Optimized LLM Configuration
# Using distilgpt2 - smallest model, fast on CPU, only ~240MB
LLM_MODEL = os.getenv("LLM_MODEL", "EleutherAI/pythia-14m")
LLM_MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", "80"))  # Shorter for speed
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.9"))
USE_LLM = os.getenv("USE_LLM", "1") == "1"  # Can disable LLM entirely

# In-memory stores
conversation_store: Dict[str, Dict] = {}
rate_limit_store: Dict[str, List[float]] = defaultdict(list)

# Global LLM components
llm_generator = None
executor = ThreadPoolExecutor(max_workers=2)  # Reduced for CPU

# OpenAPI security
security = HTTPBearer(auto_error=False)

# =========================
# LIGHTWEIGHT LLM AGENT
# =========================
class LightweightLLMAgent:
    """CPU-optimized agent using distilgpt2"""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        if not HF_AVAILABLE:
            logger.warning("Transformers not available. Using fallback responses.")
            return
        
        if not USE_LLM:
            logger.info("LLM disabled (USE_LLM=0). Using fallback responses only.")
            return
        
        logger.info(f"Initializing CPU-optimized LLM: {model_name}")
        
        try:
            # Load lightweight model optimized for CPU
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='left'
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            
            # Set model to eval mode for inference
            self.model.eval()
            
            logger.info(f"✅ LLM initialized successfully on CPU")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.model = None
    
    def _build_prompt(
        self, 
        scammer_message: str, 
        turn: int
    ) -> str:
        """Build a short, focused prompt for fast generation"""
        
        # Simple persona based on turn
        if turn <= 2:
            persona = "confused elderly person"
        elif turn <= 4:
            persona = "trusting person who wants to help"
        elif turn <= 6:
            persona = "worried person asking for details"
        else:
            persona = "person ready to act"
        
        # Short prompt for faster generation
        prompt = f"Scammer says: {scammer_message[:200]}\n\nReply as a {persona}: "
        
        return prompt
    
    def generate_response(
        self,
        scammer_message: str,
        turn: int,
        confidence: float,
        conversation_history: List[str] = None
    ) -> str:
        """Generate response using lightweight LLM"""
        
        if not self.model or not self.tokenizer:
            return self._fallback_response(turn, confidence)
        
        try:
            prompt = self._build_prompt(scammer_message, turn)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=200,  # Limit input length
                truncation=True
            )
            
            # Generate with optimized parameters for CPU
            with torch.no_grad():  # Disable gradient computation
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=LLM_MAX_LENGTH,
                    temperature=LLM_TEMPERATURE,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            response = generated_text[len(prompt):].strip()
            
            # Clean up response
            response = response.split("\n")[0].strip()  # First line only
            response = response.strip('"\'.,')
            
            # Limit length
            if len(response) > 200:
                response = response[:197] + "..."
            
            # Quality check
            if len(response) < 15 or not response:
                return self._fallback_response(turn, confidence)
            
            # Add natural variations
            response = self._add_natural_touches(response, turn)
            
            logger.info(f"LLM response (turn {turn}): {response[:80]}...")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(turn, confidence)
    
    def _add_natural_touches(self, response: str, turn: int) -> str:
        """Add natural human touches to response"""
        
        # Add question marks if missing for early turns
        if turn <= 3 and "?" not in response:
            if not response.endswith((".", "!", "?")):
                response += "?"
        
        # Add worry indicators for high urgency
        worry_words = ["worried", "confused", "scared", "nervous"]
        if turn <= 2 and not any(w in response.lower() for w in worry_words):
            if len(response) < 150:
                response += " I'm so worried!"
        
        return response
    
    def _fallback_response(self, turn: int, confidence: float) -> str:
        """High-quality fallback responses"""
        
        responses = [
            # Turn 1-2: Confused and worried
            "Oh no, is something wrong with my account? What happened?",
            "I don't understand. Can you please explain what I need to do?",
            
            # Turn 3-4: Trusting
            "Okay, I believe you. What information do you need from me?",
            "This sounds serious. Where do I send the details you need?",
            
            # Turn 5-6: Seeking specifics
            "Can you give me the exact account number or link I should use?",
            "What's the reference number? I want to make sure this is official.",
            
            # Turn 7-8: Ready to act
            "I'm ready to fix this now. What's the payment amount?",
            "Please send me your UPI ID so I can transfer immediately.",
        ]
        
        index = min(turn - 1, len(responses) - 1)
        return responses[index]

# =========================
# LIFESPAN MANAGER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global llm_generator
    
    logger.info("🍯 Honeypot API starting (CPU-Optimized)...")
    
    if AUTH_DISABLED:
        logger.warning("⚠️  Authentication DISABLED")
    
    # Initialize LLM Agent
    if HF_AVAILABLE and USE_LLM:
        logger.info("Initializing lightweight LLM agent...")
        llm_generator = LightweightLLMAgent(model_name=LLM_MODEL)
    else:
        logger.info("Using fallback responses only (no LLM)")
        llm_generator = LightweightLLMAgent.__new__(LightweightLLMAgent)
        llm_generator.model = None
        llm_generator.tokenizer = None
    
    yield
    
    logger.info("🍯 API shutting down...")
    executor.shutdown(wait=True)

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(
    title="Agentic Honeypot API - CPU Optimized",
    description="Lightweight honeypot for Render free tier",
    version="2.0.0-cpu",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST/RESPONSE MODELS
# =========================
class MessageRequest(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        class Config:
            arbitrary_types_allowed = True
    
    message: str = Field(..., description="The scam message content")
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    timestamp: Optional[str] = None
    sender: Optional[str] = None
    metadata: Optional[Dict] = Field(default_factory=dict)

class MessageResponse(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        class Config:
            arbitrary_types_allowed = True
    
    conversation_id: str
    scam_detected: bool
    confidence_score: float
    engagement: Dict
    extracted_intelligence: Dict
    agent_status: str
    agent_response: Optional[str] = None
    agent_type: str
    risk_level: str
    message_metadata: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    active_conversations: int
    auth_mode: str
    llm_status: str
    llm_model: str
    deployment: str

# =========================
# SECURITY
# =========================
def get_bearer_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    return credentials.credentials if credentials else None

def verify_api_key(token: Optional[str] = None, auth_header: Optional[str] = None):
    if AUTH_DISABLED:
        return True
    
    if token and token == API_KEY:
        return True
    
    if auth_header:
        extracted_token = auth_header.replace("Bearer", "").strip()
        if extracted_token == API_KEY:
            return True
    
    if not token and not auth_header:
        raise HTTPException(status_code=401, detail="Authorization required")
    
    raise HTTPException(status_code=401, detail="Invalid API key")

def check_rate_limit(client_ip: str, limit: int = 30, window: int = 60) -> bool:
    """Stricter rate limiting for free tier"""
    now = time.time()
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip] 
        if now - ts < window
    ]
    
    if len(rate_limit_store[client_ip]) >= limit:
        return False
    
    rate_limit_store[client_ip].append(now)
    return True

# =========================
# SCAM DETECTION
# =========================
class ScamDetector:
    SCAM_KEYWORDS = [
        "kyc", "blocked", "suspend", "verify", "urgent", "immediately",
        "click", "link", "bank", "account", "upi", "payment", "pay",
        "confirm", "update", "secure", "alert", "warning", "fee",
        "expired", "limited time", "act now", "congratulations",
        "winner", "prize", "claim", "refund", "tax", "penalty",
        "open", "verification", "transfer", "rupees", "rs"
    ]
    
    URGENCY_PHRASES = [
        "within 24 hours", "immediate action", "right now",
        "before it's too late", "final notice", "last chance"
    ]
    
    FINANCIAL_INDICATORS = [
        "send money", "transfer funds", "account number",
        "routing number", "cvv", "pin", "otp", "password", "pay"
    ]
    
    @staticmethod
    def detect(message: str) -> Tuple[bool, float, Dict]:
        message_lower = message.lower()
        details = {
            "keyword_matches": [],
            "urgency_detected": False,
            "financial_request": False,
            "suspicious_patterns": []
        }
        
        keyword_count = 0
        for keyword in ScamDetector.SCAM_KEYWORDS:
            if keyword in message_lower:
                keyword_count += 1
                details["keyword_matches"].append(keyword)
        
        for phrase in ScamDetector.URGENCY_PHRASES:
            if phrase in message_lower:
                details["urgency_detected"] = True
                break
        
        for indicator in ScamDetector.FINANCIAL_INDICATORS:
            if indicator in message_lower:
                details["financial_request"] = True
                break
        
        if re.search(r"(open|click).*link", message_lower):
            details["suspicious_patterns"].append("link_action_request")
            
        if re.search(r"pay.*(?:fee|verification|upi|account)", message_lower):
            details["suspicious_patterns"].append("payment_request")
        
        if re.search(r"verify.*(?:account|identity|information|kyc)", message_lower):
            details["suspicious_patterns"].append("verification_request")
            
        if re.search(r"\d+.*upi", message_lower):
            details["suspicious_patterns"].append("payment_amount_with_upi")
        
        confidence = 0.0
        confidence += min(0.60, keyword_count * 0.12)
        confidence += 0.25 if details["urgency_detected"] else 0.0
        confidence += 0.25 if details["financial_request"] else 0.0
        confidence += len(details["suspicious_patterns"]) * 0.15
        
        confidence = min(0.99, confidence)
        is_scam = keyword_count >= SCAM_KEYWORD_THRESHOLD or confidence >= 0.35
        
        return is_scam, confidence, details

# =========================
# INTELLIGENCE EXTRACTION
# =========================
class IntelligenceExtractor:
    @staticmethod
    def extract(text: str) -> Dict:
        bank_accounts = re.findall(r'\b\d{9,18}\b', text)
        upi_ids = re.findall(r'\b[\w.-]+@[\w.-]+\b', text)
        upi_ids = [u for u in upi_ids if not any(
            domain in u.lower() 
            for domain in ['gmail', 'yahoo', 'outlook', 'hotmail']
        )]
        
        urls = re.findall(r'https?://[^\s]+', text)
        phones = re.findall(
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            text
        )
        crypto_addresses = re.findall(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b', text)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        
        return {
            "bank_accounts": list(set(bank_accounts)),
            "upi_ids": list(set(upi_ids)),
            "phishing_urls": list(set(urls)),
            "phone_numbers": list(set(phones)),
            "crypto_addresses": list(set(crypto_addresses)),
            "email_addresses": list(set(emails))
        }

def calculate_risk_level(confidence: float, extracted_data: Dict) -> str:
    has_sensitive_data = any([
        extracted_data.get("bank_accounts"),
        extracted_data.get("upi_ids"),
        extracted_data.get("crypto_addresses")
    ])
    
    if confidence >= 0.8 or has_sensitive_data:
        return "CRITICAL"
    elif confidence >= 0.6:
        return "HIGH"
    elif confidence >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# =========================
# ASYNC LLM WRAPPER
# =========================
async def generate_agent_response_async(
    message: str,
    turn: int,
    confidence: float,
    history: List[str]
) -> Tuple[str, str]:
    """Async wrapper for LLM generation"""
    loop = asyncio.get_event_loop()
    
    if llm_generator and llm_generator.model:
        response = await loop.run_in_executor(
            executor,
            llm_generator.generate_response,
            message,
            turn,
            confidence,
            history
        )
        return response, "llm"
    else:
        response = llm_generator._fallback_response(turn, confidence) if llm_generator else "Please provide more details."
        return response, "fallback"

# =========================
# API ENDPOINTS
# =========================
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="operational",
        timestamp=datetime.utcnow().isoformat(),
        active_conversations=len(conversation_store),
        auth_mode="disabled" if AUTH_DISABLED else "enabled",
        llm_status="active" if (llm_generator and llm_generator.model) else "fallback",
        llm_model=LLM_MODEL,
        deployment="render-free-tier-cpu"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        active_conversations=len(conversation_store),
        auth_mode="disabled" if AUTH_DISABLED else "enabled",
        llm_status="active" if (llm_generator and llm_generator.model) else "fallback",
        llm_model=LLM_MODEL,
        deployment="render-free-tier-cpu"
    )

@app.post("/honeypot/message", response_model=MessageResponse)
async def process_message(
    request: MessageRequest,
    req: Request,
    token: Optional[str] = Depends(get_bearer_token),
    authorization: Optional[str] = Header(None)
):
    """Main honeypot endpoint"""
    verify_api_key(token, authorization)
    
    client_ip = req.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    start_time = time.time()
    
    conv_id = request.conversation_id or str(uuid.uuid4())
    message = request.message
    
    metadata = request.metadata.copy() if request.metadata else {}
    if request.message_id:
        metadata["message_id"] = request.message_id
    if request.timestamp:
        metadata["timestamp"] = request.timestamp
    if request.sender:
        metadata["sender"] = request.sender
    
    if conv_id not in conversation_store:
        conversation_store[conv_id] = {
            "messages": [],
            "agent_responses": [],
            "turns": 0,
            "start_time": start_time,
            "scam_detected": False,
            "confidence": 0.0,
            "detection_details": {},
            "extracted": {
                "bank_accounts": [],
                "upi_ids": [],
                "phishing_urls": [],
                "phone_numbers": [],
                "crypto_addresses": [],
                "email_addresses": []
            },
            "metadata": metadata
        }
    
    conv = conversation_store[conv_id]
    conv["messages"].append(message)
    conv["turns"] += 1
    
    # Scam detection
    is_scam, confidence, detection_details = ScamDetector.detect(message)
    
    if is_scam:
        conv["scam_detected"] = True
        conv["confidence"] = max(conv["confidence"], confidence)
        conv["detection_details"] = detection_details
    
    # Extract intelligence
    extracted = IntelligenceExtractor.extract(message)
    for key in conv["extracted"]:
        if key in extracted:
            conv["extracted"][key].extend(extracted[key])
            conv["extracted"][key] = list(set(conv["extracted"][key]))
    
    # Calculate metrics
    duration = int(time.time() - conv["start_time"])
    agent_status = "engaging" if conv["scam_detected"] and conv["turns"] < MAX_TURNS else "monitoring"
    risk_level = calculate_risk_level(conv["confidence"], conv["extracted"])
    
    # Generate response
    agent_response = None
    agent_type = "none"
    
    if conv["scam_detected"] and conv["turns"] < MAX_TURNS:
        agent_response, agent_type = await generate_agent_response_async(
            message,
            conv["turns"],
            conv["confidence"],
            conv["messages"]
        )
        conv["agent_responses"].append(agent_response)
    
    response = MessageResponse(
        conversation_id=conv_id,
        scam_detected=conv["scam_detected"],
        confidence_score=round(conv["confidence"], 2),
        engagement={
            "total_turns": conv["turns"],
            "duration_seconds": duration,
            "max_turns": MAX_TURNS
        },
        extracted_intelligence=conv["extracted"],
        agent_status=agent_status,
        agent_response=agent_response,
        agent_type=agent_type,
        risk_level=risk_level,
        message_metadata=metadata
    )
    
    return response

@app.get("/honeypot/stats")
async def get_stats(
    token: Optional[str] = Depends(get_bearer_token),
    authorization: Optional[str] = Header(None)
):
    verify_api_key(token, authorization)
    
    total_conversations = len(conversation_store)
    scam_conversations = sum(1 for c in conversation_store.values() if c["scam_detected"])
    
    return {
        "total_conversations": total_conversations,
        "scam_conversations": scam_conversations,
        "scam_rate": round(scam_conversations / total_conversations, 2) if total_conversations > 0 else 0,
        "llm_model": LLM_MODEL,
        "llm_active": bool(llm_generator and llm_generator.model),
        "deployment": "render-free-tier-cpu"
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
