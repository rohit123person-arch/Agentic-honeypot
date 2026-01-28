"""
Agentic Honeypot API - Production Ready
Detects and engages with scam messages to extract intelligence

Accepts input format:
{
    "conversation_id": "conv_001",
    "message_id": "msg_002", 
    "timestamp": "2026-01-28T10:16:20Z",
    "sender": "scammer",
    "message": "Your scam message here"
}
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
SCAM_KEYWORD_THRESHOLD = 2  # Require only 2 keywords to flag as scam

# In-memory stores
conversation_store: Dict[str, Dict] = {}
rate_limit_store: Dict[str, List[float]] = defaultdict(list)

# OpenAPI / docs security
security = HTTPBearer(auto_error=False)

# =========================
# LIFESPAN MANAGER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    if AUTH_DISABLED:
        logger.warning("⚠️  Authentication is DISABLED (HONEYPOT_DEV_DISABLE_AUTH=1). This is unsafe for production.")
    else:
        logger.info("✅ Authentication enabled. API Key required.")
    
    logger.info("🍯 Honeypot API starting...")
    yield
    logger.info("🍯 Honeypot API shutting down...")

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(
    title="Agentic Honeypot API",
    description="AI-powered honeypot for scam detection and intelligence gathering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
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
    """
    Request model that matches your input format exactly
    
    Example:
    {
        "conversation_id": "conv_001",
        "message_id": "msg_002",
        "timestamp": "2026-01-28T10:16:20Z",
        "sender": "scammer",
        "message": "Open the link and pay the verification fee of 199 to UPI."
    }
    """
    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        class Config:
            arbitrary_types_allowed = True
    
    # Required field
    message: str = Field(..., description="The scam message content to analyze")
    
    # Optional fields from your input format
    conversation_id: Optional[str] = Field(None, description="Unique conversation identifier (e.g., conv_001)")
    message_id: Optional[str] = Field(None, description="Unique message identifier (e.g., msg_002)")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")
    sender: Optional[str] = Field(None, description="Sender identifier (e.g., scammer)")
    
    # Additional metadata
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")

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
    risk_level: str
    
    # Include original metadata in response
    message_metadata: Optional[Dict] = None
    
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    active_conversations: int
    auth_mode: str

class ConversationSummary(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        class Config:
            arbitrary_types_allowed = True
    
    conversation_id: str
    total_turns: int
    duration_seconds: int
    scam_detected: bool
    confidence_score: float
    extracted_intelligence: Dict
    message_history: List[str]

# =========================
# SECURITY & RATE LIMITING
# =========================
def get_bearer_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Helper dependency for OpenAPI auth UI"""
    return credentials.credentials if credentials else None

def verify_api_key(token: Optional[str] = None, auth_header: Optional[str] = None):
    """Verify the API key from Authorization header or Bearer token"""
    # If auth is explicitly disabled for dev, allow all requests
    if AUTH_DISABLED:
        return True
    
    # Try bearer token first (from Depends)
    if token:
        if token == API_KEY:
            return True
    
    # Try Authorization header (from Header)
    if auth_header:
        # Accept: "Bearer key" or just "key"
        extracted_token = auth_header.replace("Bearer", "").strip()
        if extracted_token == API_KEY:
            return True
    
    # No valid auth provided
    if not token and not auth_header:
        logger.warning("Missing authorization")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    logger.warning(f"Invalid API key attempt")
    raise HTTPException(status_code=401, detail="Invalid API key")

def check_rate_limit(client_ip: str, limit: int = 60, window: int = 60) -> bool:
    """Simple rate limiting: limit requests per window (seconds)"""
    now = time.time()
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip] 
        if now - ts < window
    ]
    
    if len(rate_limit_store[client_ip]) >= limit:
        return False
    
    rate_limit_store[client_ip].append(now)
    return True

# =========================
# SCAM DETECTION ENGINE
# =========================
class ScamDetector:
    """Advanced scam detection with multiple strategies"""
    
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
        """
        Detect scam intent and return (is_scam, confidence, details)
        """
        message_lower = message.lower()
        details = {
            "keyword_matches": [],
            "urgency_detected": False,
            "financial_request": False,
            "suspicious_patterns": []
        }
        
        # Keyword detection
        keyword_count = 0
        for keyword in ScamDetector.SCAM_KEYWORDS:
            if keyword in message_lower:
                keyword_count += 1
                details["keyword_matches"].append(keyword)
        
        # Urgency detection
        for phrase in ScamDetector.URGENCY_PHRASES:
            if phrase in message_lower:
                details["urgency_detected"] = True
                break
        
        # Financial indicators
        for indicator in ScamDetector.FINANCIAL_INDICATORS:
            if indicator in message_lower:
                details["financial_request"] = True
                break
        
        # Pattern detection
        if re.search(r"(open|click).*link", message_lower):
            details["suspicious_patterns"].append("link_action_request")
            
        if re.search(r"pay.*(?:fee|verification|upi|account)", message_lower):
            details["suspicious_patterns"].append("payment_request")
        
        if re.search(r"verify.*(?:account|identity|information|kyc)", message_lower):
            details["suspicious_patterns"].append("verification_request")
            
        # Check for number + UPI pattern
        if re.search(r"\d+.*upi", message_lower):
            details["suspicious_patterns"].append("payment_amount_with_upi")
        
        # Calculate confidence score (more aggressive)
        confidence = 0.0
        confidence += min(0.60, keyword_count * 0.12)  # Up to 60% for keywords (increased)
        confidence += 0.25 if details["urgency_detected"] else 0.0  # Increased from 0.20
        confidence += 0.25 if details["financial_request"] else 0.0  # Increased from 0.20
        confidence += len(details["suspicious_patterns"]) * 0.15  # Increased from 0.10
        
        confidence = min(0.99, confidence)  # Cap at 99%
        
        # Lower threshold: flag as scam if confidence >= 0.35 OR keywords >= 2
        is_scam = keyword_count >= SCAM_KEYWORD_THRESHOLD or confidence >= 0.35
        
        return is_scam, confidence, details

# =========================
# INTELLIGENCE EXTRACTION
# =========================
class IntelligenceExtractor:
    """Extract actionable intelligence from messages"""
    
    @staticmethod
    def extract(text: str) -> Dict:
        """Extract various intelligence artifacts"""
        
        # Bank account numbers (9-18 digits)
        bank_accounts = re.findall(r'\b\d{9,18}\b', text)
        
        # UPI IDs (format: something@something)
        upi_ids = re.findall(r'\b[\w.-]+@[\w.-]+\b', text)
        # Filter out likely email domains
        upi_ids = [u for u in upi_ids if not any(
            domain in u.lower() 
            for domain in ['gmail', 'yahoo', 'outlook', 'hotmail']
        )]
        
        # URLs
        urls = re.findall(r'https?://[^\s]+', text)
        
        # Phone numbers (various formats)
        phones = re.findall(
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            text
        )
        
        # Cryptocurrency addresses (simplified)
        crypto_addresses = re.findall(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b', text)
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        
        return {
            "bank_accounts": list(set(bank_accounts)),
            "upi_ids": list(set(upi_ids)),
            "phishing_urls": list(set(urls)),
            "phone_numbers": list(set(phones)),
            "crypto_addresses": list(set(crypto_addresses)),
            "email_addresses": list(set(emails))
        }

# =========================
# AGENT BEHAVIOR
# =========================
class AgentPersona:
    """Simulates a vulnerable user to extract more intelligence"""
    
    RESPONSES = [
        # Early engagement
        "Oh no, really? What should I do? I'm worried now.",
        "I don't understand. Can you explain what's wrong with my account?",
        
        # Building trust
        "Okay, I trust you. What do you need from me?",
        "This sounds serious. Where should I send the information?",
        
        # Information gathering
        "Can you send me the official link or reference number?",
        "What details do you need exactly? Should I share my account number?",
        
        # Urgency response
        "I want to fix this immediately. What's the fastest way?",
        "Please share the exact steps. I'll do it right now.",
        
        # Final extraction
        "Okay, I'm ready. What account should I transfer to?",
        "I have my card ready. Where do I enter the details?",
    ]
    
    @staticmethod
    def generate_response(turn: int, scam_confidence: float, extracted_data: Dict) -> str:
        """Generate contextual response based on conversation state"""
        
        # More aggressive engagement for high confidence scams
        if scam_confidence > 0.7:
            urgency_boost = min(2, int(scam_confidence * 3))
            turn = min(turn + urgency_boost, len(AgentPersona.RESPONSES) - 1)
        
        # Add context based on extracted data
        response = AgentPersona.RESPONSES[min(turn, len(AgentPersona.RESPONSES) - 1)]
        
        # Occasionally ask for specific information
        if turn > 3 and not any(extracted_data.values()):
            response += " Can you share the account details or link?"
        
        return response

# =========================
# RISK ASSESSMENT
# =========================
def calculate_risk_level(confidence: float, extracted_data: Dict) -> str:
    """Calculate overall risk level"""
    
    # Check if we've extracted sensitive data
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
# MAIN API ENDPOINTS
# =========================
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="operational",
        timestamp=datetime.utcnow().isoformat(),
        active_conversations=len(conversation_store),
        auth_mode="disabled" if AUTH_DISABLED else "enabled"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        active_conversations=len(conversation_store),
        auth_mode="disabled" if AUTH_DISABLED else "enabled"
    )

@app.get("/status")
def status():
    """Simple status endpoint to inspect auth mode"""
    return {
        "auth_disabled": AUTH_DISABLED,
        "api_key_present": bool(API_KEY),
        "active_conversations": len(conversation_store)
    }

@app.post("/honeypot/message", response_model=MessageResponse)
async def process_message(
    request: MessageRequest,
    req: Request,
    token: Optional[str] = Depends(get_bearer_token),
    authorization: Optional[str] = Header(None)
):
    """
    Main honeypot endpoint - processes messages and returns analysis
    
    Accepts your input format:
    {
        "conversation_id": "conv_001",
        "message_id": "msg_002",
        "timestamp": "2026-01-28T10:16:20Z",
        "sender": "scammer",
        "message": "Open the link and pay the verification fee of 199 to UPI."
    }
    """
    # Security checks
    verify_api_key(token, authorization)
    
    client_ip = req.client.host
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    start_time = time.time()
    
    # Get or create conversation
    conv_id = request.conversation_id or str(uuid.uuid4())
    message = request.message
    
    # Build metadata from your input fields
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
        logger.info(f"New conversation started: {conv_id}")
    
    conv = conversation_store[conv_id]
    conv["messages"].append(message)
    conv["turns"] += 1
    
    # Scam detection
    is_scam, confidence, detection_details = ScamDetector.detect(message)
    
    if is_scam:
        conv["scam_detected"] = True
        conv["confidence"] = max(conv["confidence"], confidence)
        conv["detection_details"] = detection_details
        logger.warning(f"Scam detected in {conv_id} - Confidence: {confidence:.2f}")
    
    # Extract intelligence
    extracted = IntelligenceExtractor.extract(message)
    for key in conv["extracted"]:
        if key in extracted:
            conv["extracted"][key].extend(extracted[key])
            conv["extracted"][key] = list(set(conv["extracted"][key]))
    
    # Log extracted intelligence
    if any(extracted.values()):
        logger.info(f"Intelligence extracted from {conv_id}: {extracted}")
    
    # Calculate metrics
    duration = int(time.time() - conv["start_time"])
    agent_status = "engaging" if conv["scam_detected"] and conv["turns"] < MAX_TURNS else "monitoring"
    risk_level = calculate_risk_level(conv["confidence"], conv["extracted"])
    
    # Generate agent response if scam detected
    agent_response = None
    if conv["scam_detected"] and conv["turns"] < MAX_TURNS:
        agent_response = AgentPersona.generate_response(
            conv["turns"], 
            conv["confidence"],
            conv["extracted"]
        )
        conv["agent_responses"].append(agent_response)
        logger.info(f"Agent response generated for {conv_id}")
    
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
        risk_level=risk_level,
        message_metadata=metadata  # Include your original metadata in response
    )
    
    return response

@app.get("/honeypot/conversation/{conversation_id}", response_model=ConversationSummary)
async def get_conversation(
    conversation_id: str,
    token: Optional[str] = Depends(get_bearer_token),
    authorization: Optional[str] = Header(None)
):
    """
    Retrieve full conversation history and analysis
    """
    verify_api_key(token, authorization)
    
    if conversation_id not in conversation_store:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv = conversation_store[conversation_id]
    duration = int(time.time() - conv["start_time"])
    
    return ConversationSummary(
        conversation_id=conversation_id,
        total_turns=conv["turns"],
        duration_seconds=duration,
        scam_detected=conv["scam_detected"],
        confidence_score=round(conv["confidence"], 2),
        extracted_intelligence=conv["extracted"],
        message_history=conv["messages"]
    )

@app.delete("/honeypot/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    token: Optional[str] = Depends(get_bearer_token),
    authorization: Optional[str] = Header(None)
):
    """
    Delete a conversation from memory
    """
    verify_api_key(token, authorization)
    
    if conversation_id not in conversation_store:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversation_store[conversation_id]
    logger.info(f"Conversation deleted: {conversation_id}")
    
    return {"status": "deleted", "conversation_id": conversation_id}

@app.get("/honeypot/stats")
async def get_stats(
    token: Optional[str] = Depends(get_bearer_token),
    authorization: Optional[str] = Header(None)
):
    """
    Get overall honeypot statistics
    """
    verify_api_key(token, authorization)
    
    total_conversations = len(conversation_store)
    scam_conversations = sum(1 for c in conversation_store.values() if c["scam_detected"])
    
    total_intelligence = {
        "bank_accounts": 0,
        "upi_ids": 0,
        "phishing_urls": 0,
        "phone_numbers": 0,
        "crypto_addresses": 0,
        "email_addresses": 0
    }
    
    for conv in conversation_store.values():
        for key in total_intelligence:
            total_intelligence[key] += len(conv["extracted"].get(key, []))
    
    avg_confidence = (
        sum(c["confidence"] for c in conversation_store.values()) / total_conversations
        if total_conversations > 0 else 0
    )
    
    return {
        "total_conversations": total_conversations,
        "scam_conversations": scam_conversations,
        "scam_rate": round(scam_conversations / total_conversations, 2) if total_conversations > 0 else 0,
        "average_confidence": round(avg_confidence, 2),
        "total_intelligence_extracted": total_intelligence,
        "active_rate_limits": len(rate_limit_store)
    }

# =========================
# ERROR HANDLERS
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error handler"""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors"""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
