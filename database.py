from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# URL สำหรับเชื่อมต่อ MySQL
# แทนค่า: username, password, host, dbname
DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/myproject"

# สร้าง engine
engine = create_engine(DATABASE_URL, echo=True)

# SessionLocal ใช้สร้าง session เวลาจะคุยกับ DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base สำหรับสร้าง Models
Base = declarative_base()
