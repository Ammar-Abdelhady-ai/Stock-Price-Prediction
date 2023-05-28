import yfinance as yf
from datetime import datetime






yf.pdr_override()

############################################
end = datetime.now()
start = datetime(end.year - 11, end.month, end.day)

#############################################################################
apple_df = yf.download("AAPL", start, end)
describe = apple_df.describe()
print(describe)
