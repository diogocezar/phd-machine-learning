from scipy.stats import norm

p_yes_outlook_rainy = 4/12
p_yes_temp_65 = norm(72.9697, 5.2304).pdf(65)
p_yes_humidity_70 = norm(78.8395, 9.8023).pdf(70)
p_yes_wind = 4/11
p_yes_class = 0.63

# P(Yes|E)
p_yes_e = p_yes_outlook_rainy * \
    p_yes_temp_65 * \
    p_yes_humidity_70 * \
    p_yes_wind * \
    p_yes_class

print("p_yes_e = ", p_yes_e)

p_no_outlook_rainy = 3/8
p_no_temp_65 = norm(74.8364, 7.384).pdf(65)
p_no_humidity_70 = norm(86.1111, 9.2424).pdf(70)
p_no_wind = 4/7
p_no_class = 0.38

# P(No|E)
p_no_e = p_no_outlook_rainy * \
    p_no_temp_65 * \
    p_no_humidity_70 * \
    p_no_wind * \
    p_no_class

print("p_no_e = ", p_no_e)

p_yes_e_norm = p_yes_e / (p_yes_e + p_no_e)
p_no_e_norm = p_no_e / (p_yes_e + p_no_e)

print("p_yes_e_norm = ", p_yes_e_norm)
print("p_no_e_norm = ", p_no_e_norm)
