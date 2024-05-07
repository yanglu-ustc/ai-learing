import numpy as np

data = np.genfromtxt('Main.csv', delimiter='&')
print(data)
r_big = np.mean(data[0]) * (10 ** (-2))
print(r_big)
m_big = np.mean(data[1]) * (10 ** (-3))
print(m_big)
t_big = np.mean(data[2])
print(t_big)
r_mo = np.mean(data[3]) * (10 ** (-2))
print(r_mo)
m_mo = np.mean(data[4]) * (10 ** (-3))
print(m_mo)
t_mo = np.mean(data[5])
print(t_mo)
r_small = np.mean(data[6]) * (10 ** (-2))
print(r_small)
m_small = np.mean(data[7]) * (10 ** (-3))
print(m_small)
t_small = np.mean(data[8])
print(t_small)

print("\n" * 2)

l = 18.75 * (10 ** (-2))
v_big = l / t_big
print(v_big)
v_mo = l / t_mo
print(v_mo)
v_small = l / t_small
print(v_small)

print("\n" * 2)

data = np.genfromtxt('qita.csv', delimiter='&')
print(data)
rou = np.mean(data[0]) * (10 ** (3))
print(rou)
T = np.mean(data[1])
print(T)
D = np.mean(data[2]) * (10 ** (-2))
print(D)
h = np.mean(data[3]) * (10 ** (-2))
print(h)

print("\n" * 2)

yita_big = (m_big - rou * (4 * np.pi / 3) * (r_big ** 3)) * 9.8 / (6 * np.pi * v_big * r_big)
print(yita_big)
yita_mo = (m_mo - rou * (4 * np.pi / 3) * (r_mo ** 3)) * 9.8 / (6 * np.pi * v_mo * r_mo)
print(yita_mo)
yita_small = (m_small - rou * (4 * np.pi / 3) * (r_small ** 3)) * 9.8 / (6 * np.pi * v_small * r_small)
print(yita_small)

print("\n" * 2)

re_big = 2 * r_big * rou * v_big / yita_big
print(re_big)
re_mo = 2 * r_mo * rou * v_mo / yita_mo
print(re_mo)
re_small = 2 * r_small * rou * v_small / yita_small
print(re_small)

print("\n" * 2)


def rou_(m, r):
    return 3 * m / (4 * np.pi * (r ** 3))


def yita_0(m, r, v):
    result = (1 / 18) * (rou_(m, r) - rou) * 9.8 * ((2 * r) ** 2) / ((1 + 2.4 * 2 * r / D) * v * (1 + 3.3 * r / h))
    return result


def yita_1(m, r, v):
    return yita_0(m, r, v) - 3 * 2 * r * rou * v / 16


def yita_2(m, r, v):
    return (yita_1(m, r, v) / 2) * (1 + np.sqrt(1 + (19 / 270) * ((2 * r * rou * v / yita_1(m, r, v)) ** 2)))


def yita_my(m, r, v, re):
    return yita_0(m, r, v) / (1 + 3 * re / 16)


print("==="*99)


print(yita_big)
print(yita_0(m_big, r_big, v_big))
print(yita_1(m_big, r_big, v_big))
print(yita_2(m_big, r_big, v_big))
print(yita_my(m_big, r_big, v_big, re_big))
print("\n" * 2)

print(yita_mo)
print(yita_0(m_mo, r_mo, v_mo))
print(yita_1(m_mo, r_mo, v_mo))
print(yita_2(m_mo, r_mo, v_mo))
print(yita_my(m_mo, r_mo, v_mo, re_mo))
print("\n" * 2)

print(yita_small)
print(yita_0(m_small, r_small, v_small))
print(yita_1(m_small, r_small, v_small))
print(yita_2(m_small, r_small, v_small))
print(yita_my(m_small, r_small, v_small, re_small))
