def stenosis_category(percent):
    percent = float(percent)
    if percent == 100:
        return '13'
    elif 90 <= percent <= 99:
        return '12'
    elif 80 <= percent <= 99:
        return '11'
    elif 80 <= percent <= 89:
        return '10'
    elif 70 <= percent <= 89:
        return '9'
    elif 70 <= percent <= 80:
        return '8'
    elif 65 <= percent <= 70:
        return '7'
    elif 60 <= percent <= 89:
        return '6'
    elif 60 <= percent <= 79:
        return '5'
    elif 50 <= percent <= 69:
        return '4'
    elif 50 <= percent <= 59:
        return '3'
    elif 0 <= percent <= 69:
        return '2'
    else:
        return '1'
