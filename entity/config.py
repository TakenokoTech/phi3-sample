from dataclasses import dataclass


@dataclass
class Config:
    k = 3
    p = 0.95
    temperature = 0.99
