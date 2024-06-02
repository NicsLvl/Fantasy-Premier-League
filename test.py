coffee = ['coldbrew', 'drip', 'americano']
milk = ['whole', 'skim', 'almond']
temp = ['hot', 'iced', 'hot']


def is_drip(coffee, milk, temp):
    return 'drip' in coffee and 'skim' in milk and 'iced' in temp


agg = filter(lambda value: is_drip(*value), zip(coffee, milk, temp))

drip_orders = list(agg)

print(drip_orders)
