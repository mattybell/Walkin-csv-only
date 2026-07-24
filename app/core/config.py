from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # IMPORTANT: your docker maps host 55433 -> container 5432
    DATABASE_URL: str = "postgresql+psycopg://walkin:walkin@127.0.0.1:55433/walkin"

    JWT_SECRET: str = "dev-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    PROOF_TOKEN_TTL_SECONDS: int = 60 * 5  # 5 minutes

    # Free geocoding for dev. For production, use a paid provider + key.
    NOMINATIM_USER_AGENT: str = "walkin-dev"

    # When True: emails and SMS are printed to console instead of sent.
    # Set LOCAL_MODE=true in .env for local development.
    LOCAL_MODE: bool = False

    # Twilio credentials for SMS OTP (phone auth).
    # Add these to backend/.env or GitLab CI/CD variables.
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""

    # SMTP credentials for transactional email (email verification).
    # Add these to backend/.env for local testing.
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "noreply@walkin.to"
    
    # External API Keys
    GOOGLE_PLACES_API_KEY: str = ""  # Google Places API for business import
    BRIGHTDATA_API_KEY: str = ""  # BrightData API for location popularity data
    # BrightData Datasets API dataset id for the Google Maps business scraper
    # (returns popular_times among other fields). See backend/.env.example.
    BRIGHTDATA_DATASET_ID: str = "gd_m8ebnr0q2qlklc02fz"

    # Stripe billing (merchant subscription gate).
    # Add these to backend/.env or GitLab CI/CD variables.
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_ID: str = ""  # Stripe Price object id for the $119/month plan
    LANDING_PAGE_URL: str = "https://walkin.to"
    # Comma-separated hostnames allowed to serve the landing page HTML (walkin.to root
    # domain). api.secure.walkin.to shares this same backend service but must NOT serve
    # the landing page — that hostname is mobile-API-only.
    MARKETING_HOSTNAMES: str = "walkin.to,www.walkin.to"

    # Logging
    LOG_LEVEL: str = "INFO"


# Instantiate settings (loads from .env)
settings = Settings()
