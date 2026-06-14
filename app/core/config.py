from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Microservice"
    DATABASE_URL: str = "mysql+pymysql://user:password@localhost:3306/db_name"
    
    class Config:
        env_file = ".env"

settings = Settings()
