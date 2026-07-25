import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)

    # "customer" | "business"
    role: Mapped[str] = mapped_column(String, nullable=False, default="customer", index=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Minimal customer profile fields (optional)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    address: Mapped[str | None] = mapped_column(String, nullable=True)
    gender: Mapped[str | None] = mapped_column(String, nullable=True)  # "male", "female", "other", "prefer_not_to_say"
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Customer's preferred radius for deal discovery (in miles). Default 10 miles.
    preferred_radius_miles: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    
    # Customer's coordinates for location-based deals (geocoded from address)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Optional avatar image (served from /static)
    avatar_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Email verification
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_verification_token_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    email_verification_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Phone OTP (for phone-based login)
    phone_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    phone_otp_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    phone_otp_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    merchant_profile = relationship("MerchantProfile", back_populates="user", uselist=False)


class Business(Base):
    __tablename__ = "businesses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)

    address_primary: Mapped[str | None] = mapped_column(String, nullable=True)
    address_secondary: Mapped[str | None] = mapped_column(String, nullable=True)

    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Google Places ID - canonical identifier for the business
    # This is the source of truth for business identity and location data
    place_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True, index=True)
    
    # Phone number (from Google Places or merchant-provided)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)

    # Business website URL (from Google Places or merchant-provided)
    website: Mapped[str | None] = mapped_column(String, nullable=True)

    # Business contact email — null on import, set to claimant's signup email on claim
    email: Mapped[str | None] = mapped_column(String, nullable=True)

    # Regular opening hours (from Google Places), stored as plain text — one line per day
    # e.g. "Monday: 9 AM – 9 PM\nTuesday: 9 AM – 9 PM\n..."
    opening_hours: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Business category/type (restaurant, cafe, gym, spa, salon, etc.)
    category: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # URL to uploaded logo for this business
    # Optional: businesses may not have a logo initially.
    logo_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Optional banner/cover image URL for this business (served from /static)
    banner_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Business-provided average spend per customer (in dollars). Used to estimate potential earnings.
    avg_spend_per_customer: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Rating and review fields
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)  # Average rating 0-5
    rating_count: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)  # Number of reviews
    about_text: Mapped[str | None] = mapped_column(String(150), nullable=True)  # Short description

    # Admin approval status
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Data source: "created" (user-created) or "osm" (imported from OpenStreetMap)
    source: Mapped[str] = mapped_column(String, default="created", nullable=False, index=True)

    # Timestamp when business was claimed by a merchant; null if unclaimed
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    deals = relationship("Deal", back_populates="business")
    merchant_profile = relationship("MerchantProfile", back_populates="business", uselist=False)
    reviews = relationship("BusinessReview", back_populates="business")


class MerchantProfile(Base):
    __tablename__ = "merchant_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), unique=True, index=True)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id"), unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="merchant_profile")
    business = relationship("Business", back_populates="merchant_profile")


class Deal(Base):
    __tablename__ = "deals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id"), index=True, nullable=False)

    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # naive UTC timestamps for now
    start_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Optional: merchant-provided estimate of value per redemption
    estimated_value_per_redemption: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Rate limiting values.  A customer cannot redeem this deal
    # more often than once per cooldown period, and may have a maximum number of redemptions
    # per 24‑hour period. cooldown_hours is mandatory.
    cooldown_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_per_day: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # V3: deal type and meta
    # Type of the deal.  "STANDARD" means a generic deal with no quantity-based logic.
    # "BUY_X_GET_Y" means the customer must buy `buy_qty` items to get `get_qty` free items.  See
    # `buy_qty` and `get_qty` for associated values.  Additional deal types can be added in
    # the future as needed.
    deal_type: Mapped[str] = mapped_column(String, nullable=False, default="STANDARD")
    # Number of items a customer must buy when deal_type == "BUY_X_GET_Y".  Null otherwise.
    buy_qty: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Number of items a customer receives for free when deal_type == "BUY_X_GET_Y".  Null otherwise.
    get_qty: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Comma-separated list of day abbreviations (e.g. "Mon,Tue,Fri").  If not null,
    # this deal can only be redeemed on the specified days.  Deals remain visible to
    # customers regardless of repeat_days.  Null means redeemable on any day.
    repeat_days: Mapped[str | None] = mapped_column(String, nullable=True)

    # V6: richer deal types (values are nullable; validation happens at the API layer)
    percent_off: Mapped[int | None] = mapped_column(Integer, nullable=True)
    item_name: Mapped[str | None] = mapped_column(String, nullable=True)
    fixed_price_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_spend_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    amount_off_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    custom_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Image URL for deal card background
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Featured deal timestamp (null if not featured)
    featured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Loyalty rewards: number of redemptions required to unlock a reward
    loyalty_redemptions_required: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Description of the reward (e.g., "Free item on 6th visit")
    loyalty_reward_description: Mapped[str | None] = mapped_column(String, nullable=True)
    
    # Progressive deals: allow multiple redemptions with unique codes each time
    is_progressive: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    business = relationship("Business", back_populates="deals")
    redemptions = relationship("Redemption", back_populates="deal")
    customer_progress = relationship("CustomerDealProgress", back_populates="deal")


class Redemption(Base):
    __tablename__ = "redemptions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    deal_id: Mapped[str | None] = mapped_column(String, ForeignKey("deals.id"), index=True, nullable=True)
    business_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("businesses.id"), index=True, nullable=True
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    device_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    device_lng: Mapped[float | None] = mapped_column(Float, nullable=True)

    # A short, human‑friendly one‑time code shown to the customer at redemption time.
    # This code is used by merchants to reconcile redemptions offline.  It is unique
    # per customer and deal at the time of creation to avoid collisions.
    otp_code: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # The moment when this redemption expires.  Merchants must validate proof tokens
    # before this time; after expiry the proof is considered invalid.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # When the merchant actually verified and accepted this redemption.  Null if
    # the redemption has not yet been reconciled by the merchant.
    redeemed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # How the merchant reconciled this redemption.
    # "QR" when a proof token was scanned/validated; "CODE" when an 8-digit
    # redemption code was entered.
    verified_method: Mapped[str | None] = mapped_column(String, nullable=True)

    deal = relationship("Deal", back_populates="redemptions")


class CustomerDealProgress(Base):
    """
    Tracks a customer's progress toward loyalty rewards for a specific deal.
    Records how many times a customer has redeemed a deal and when they earned rewards.
    """
    __tablename__ = "customer_deal_progress"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    deal_id: Mapped[str] = mapped_column(String, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)

    # Number of times this customer has successfully redeemed this deal
    redemption_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # When the loyalty reward was earned (if applicable)
    reward_earned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    deal = relationship("Deal", back_populates="customer_progress")

    __table_args__ = (
        # Ensure one record per customer per deal
        # Each customer can have at most one progress record per deal
    )


class BusinessReview(Base):
    __tablename__ = "business_reviews"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)  # Optional text review (max 500 chars)

    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Customer redeemed a deal
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Moderation flag
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)  # Admin moderation

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    business = relationship("Business", back_populates="reviews")
    customer = relationship("User", foreign_keys=[customer_id])


class Favorite(Base):
    __tablename__ = "favorites"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Ensure each user can only favorite a business once
        # Use a unique constraint on (user_id, business_id)
        # This is handled by the database constraint level for efficiency
    )

    user = relationship("User", foreign_keys=[user_id])
    business = relationship("Business", foreign_keys=[business_id])


class BusinessReviewComment(Base):
    __tablename__ = "business_review_comments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    review_id: Mapped[str] = mapped_column(String, ForeignKey("business_reviews.id", ondelete="CASCADE"), nullable=False, index=True)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)
    merchant_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    comment: Mapped[str] = mapped_column(Text, nullable=False)  # Merchant's response to the review

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    review = relationship("BusinessReview", foreign_keys=[review_id])
    business = relationship("Business", foreign_keys=[business_id])
    merchant = relationship("User", foreign_keys=[merchant_id])


class FavoriteDeal(Base):
    __tablename__ = "favorite_deals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    deal_id: Mapped[str] = mapped_column(String, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", foreign_keys=[user_id])
    deal = relationship("Deal", foreign_keys=[deal_id])


# ========== ML & Foot Traffic Models ==========


class LocationPing(Base):
    """Stores anonymized user location pings for foot traffic analysis."""
    __tablename__ = "location_pings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    
    # H3 cell for spatial indexing (resolution 9 = ~0.1km)
    h3_cell: Mapped[str] = mapped_column(String, nullable=False, index=True)
    
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Optional: accuracy in meters
    accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    user = relationship("User", foreign_keys=[user_id])


class FootTrafficAggregate(Base):
    """Hourly aggregates of foot traffic by H3 cell."""
    __tablename__ = "foot_traffic_aggregates"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    
    h3_cell: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)  # hourly buckets
    
    unique_users: Mapped[int] = mapped_column(Integer, nullable=False)
    total_pings: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Day of week (0=Monday, 6=Sunday)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    
    # Hour of day (0-23)
    hour: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DealPerformanceSnapshot(Base):
    """Historical performance snapshots of deals for ML training."""
    __tablename__ = "deal_performance_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    deal_id: Mapped[str] = mapped_column(String, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)
    
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Traffic metrics
    estimated_foot_traffic: Mapped[int] = mapped_column(Integer, nullable=False)  # from H3 cells
    actual_views: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    actual_redemptions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Deal attributes at time of snapshot
    deal_type: Mapped[str] = mapped_column(String, nullable=False)  # "BOGO", "Discount", etc.
    discount_value: Mapped[float] = mapped_column(Float, nullable=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    # Temporal features
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    # Business context
    business_category: Mapped[str] = mapped_column(String, nullable=False)
    business_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    deal = relationship("Deal", foreign_keys=[deal_id])
    business = relationship("Business", foreign_keys=[business_id])


class MLModel(Base):
    """Tracks ML model versions and metadata."""
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    
    model_type: Mapped[str] = mapped_column(String, nullable=False, index=True)  # "foot_traffic", "conversion"
    version: Mapped[str] = mapped_column(String, nullable=False)
    
    # Model file path (relative to backend/models/)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    
    # Training metrics
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Training metadata
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    training_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Hyperparameters (JSON stored as text)
    hyperparameters: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DealForecast(Base):
    """Cached predictions for deals."""
    __tablename__ = "deal_forecasts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    deal_id: Mapped[str] = mapped_column(String, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Forecast metrics
    estimated_views: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_redemptions: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_revenue: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Confidence interval
    confidence_level: Mapped[str] = mapped_column(String, nullable=False)  # "high", "medium", "low"
    redemptions_low: Mapped[int] = mapped_column(Integer, nullable=False)
    redemptions_high: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Best times
    peak_hours: Mapped[str] = mapped_column(String, nullable=False)  # JSON array: [12, 13, 18, 19]
    best_days: Mapped[str] = mapped_column(String, nullable=False)  # JSON array: ["Friday", "Saturday"]
    
    # Model version used
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    
    # Forecast validity
    forecast_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    valid_until: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    deal = relationship("Deal", foreign_keys=[deal_id])


class LocationPopularity(Base):
    """Stores location popularity data from BrightData or other sources.
    
    This is OPTIONAL enrichment data keyed by Google Place ID.
    The app functions normally without this data.
    """
    __tablename__ = "location_popularity"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    business_id: Mapped[str] = mapped_column(String, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Optional: Google Place ID for linking enrichment data
    # BrightData enrichment is keyed by this canonical identifier
    place_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    
    # Temporal info
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False, index=True)  # 0-23
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)  # 0=Monday, 6=Sunday
    
    # Date of data (for tracking when data was collected)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Popularity score (number of visitors or normalized score)
    popularity_score: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Source of data ("brightdata", "location_pings", "manual", etc.)
    source: Mapped[str] = mapped_column(String, default="brightdata", nullable=False)
    
    # When this data was fetched
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    business = relationship("Business", foreign_keys=[business_id])


# ========== Merchant Subscription (Stripe) Models ==========


class MerchantSubscription(Base):
    """Tracks whether an email has an active WalkIn Business subscription.

    Keyed by email rather than user_id because the subscription must exist
    *before* a business account does — a merchant subscribes on the landing
    page first, then the signup gate checks this table by email.
    """
    __tablename__ = "merchant_subscriptions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)

    stripe_customer_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # Mirrors Stripe's own subscription status strings directly:
    # "incomplete" | "trialing" | "active" | "past_due" | "canceled" | "unpaid"
    status: Mapped[str] = mapped_column(String, nullable=False, default="incomplete", index=True)

    current_period_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # "stripe" | "trial_code" | "manual" (manual = admin hand-granted)
    source: Mapped[str] = mapped_column(String, nullable=False, default="stripe")
    trial_code_used: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TrialCode(Base):
    """One-month free trial promo codes (30 issued at launch)."""
    __tablename__ = "trial_codes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)

    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    used_by_email: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

