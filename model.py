from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "user"
    userId = Column(String(5), primary_key=True)
    username = Column(String(50), nullable=False)
    password = Column(String(20), nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    address = Column(String(255))

    consultations = relationship("Consultation", back_populates="user")


class Expert(Base):
    __tablename__ = "expert"
    expertId = Column(String(5), primary_key=True)
    expertName = Column(String(50), nullable=False)
    password = Column(String(20), nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    address = Column(String(255))

    advices = relationship("Advice", back_populates="expert")


class Consultation(Base):
    __tablename__ = "consultation"
    consultationId = Column(String(5), primary_key=True)
    Mresult = Column(Text, nullable=False)
    dateTime = Column(DateTime)
    image = Column(Text, nullable=False)
    message = Column(Text)
    status = Column(String(50), nullable=False)
    userId = Column(String(255), ForeignKey("user.userId"))

    user = relationship("User", back_populates="consultations")
    advices = relationship("Advice", back_populates="consultation")
    plant = relationship("Plant", back_populates="consultation")


class Advice(Base):
    __tablename__ = "advice"
    adviceId = Column(String(5), primary_key=True)
    message = Column(Text)
    adDateTime = Column(DateTime)
    consultationId = Column(String(255), ForeignKey("consultation.consultationId"))
    expertId = Column(String(255), ForeignKey("expert.expertId"))

    consultation = relationship("Consultation", back_populates="advices")
    expert = relationship("Expert", back_populates="advices")


class Plant(Base):
    __tablename__ = "plant"
    plantId = Column(String(5), primary_key=True)
    care = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    picture = Column(Text)
    plantName = Column(String(50), nullable=False)
    plantType = Column(String(50), nullable=False)
    prepare = Column(Text, nullable=False)
    consultation_consultationId = Column(String(255), ForeignKey("consultation.consultationId"))

    consultation = relationship("Consultation", back_populates="plant")
    diseases = relationship("Disease", back_populates="plant")
    plantings = relationship("Planting", back_populates="plant")


class Planting(Base):
    __tablename__ = "planting"
    plantingId = Column(String(6), primary_key=True)
    plantingMethod = Column(Text, nullable=False)
    plantId = Column(String(255), ForeignKey("plant.plantId"))

    plant = relationship("Plant", back_populates="plantings")


class Disease(Base):
    __tablename__ = "disease"
    diseaseId = Column(String(5), primary_key=True)
    diseaseName = Column(String(255), nullable=False)
    symptoms = Column(Text, nullable=False)
    plantId = Column(String(255), ForeignKey("plant.plantId"))

    plant = relationship("Plant", back_populates="diseases")
    treatments = relationship("Treatment", back_populates="disease")
    medicines = relationship("Medicine", back_populates="disease")


class Treatment(Base):
    __tablename__ = "treatment"
    treatmentId = Column(String(5), primary_key=True)
    treatmentMethods = Column(Text, nullable=False)
    diseaseId = Column(String(255), ForeignKey("disease.diseaseId"))

    disease = relationship("Disease", back_populates="treatments")


class Medicine(Base):
    __tablename__ = "medicine"
    medicineId = Column(String(5), primary_key=True)
    dosage = Column(Integer)
    quantityType = Column(String(50))
    medicineName = Column(String(255), nullable=False)
    diseaseId = Column(String(255), ForeignKey("disease.diseaseId"))

    disease = relationship("Disease", back_populates="medicines")
