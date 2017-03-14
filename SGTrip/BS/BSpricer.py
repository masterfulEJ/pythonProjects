# -*- coding: utf-8 -*-

import math as m
from scipy.stats import norm
from scipy.linalg import solve_banded
from scipy.optimize import root
import numpy as np
import copy

# Black and Scholes option model
class PayoffCall(object):
    def __init__(self, strike):
        self.strike = strike

    def __call__(self, x):
        return max(x - self.strike, 0)


class PayoffPut(PayoffCall):
    def __call__(self, x):
        return max(self.strike - x, 0)


class ModelBlackScholes(object):
    def __init__(self, s, expiry, payoff, r=0.0, sigma=0.0, y=0.0, sigma_surface=None, delta=None, moneyness=None):
        self.s = s
        self.expiry = expiry
        self.payoff = payoff
        self.r = r
        self.sigma = sigma
        self.y = y
        self.sigma_surface = sigma_surface
        self.sigma_update()
        self.strike_update(delta, moneyness)

    def sigma_update(self):
        if self.sigma_surface is not None:
            self.sigma = self.sigma_surface(self.expiry, self.payoff.strike / self.s)

    def strike_update(self, delta=None, moneyness=None):
        if delta is not None:
            self.set_strike(self.s)
            self.set_strike(self.strike_from_delta(delta))
        if moneyness is not None:
            self.set_strike(moneyness * self.s)

    def d1(self):
        try:
            if self.sigma == 0:
                raise ZeroDivisionError
            d = (m.log(self.s / self.payoff.strike) + (
                self.r - self.y + self.sigma ** 2 / 2) * self.expiry) / self.sigma / m.sqrt(
                self.expiry)
        except ZeroDivisionError:
            d = np.inf * (1.0 if self.s > self.payoff.strike else -1.0)
        return d

    def d2(self):
        return self.d1() - self.sigma * m.sqrt(self.expiry)

    def __call__(self):
        if type(self.payoff) in [PayoffCall]:
            price = self.s * m.exp(-self.y * self.expiry) * norm.cdf(self.d1()) - self.payoff.strike * m.exp(
                -self.r * self.expiry) * norm.cdf(
                self.d2())
        elif type(self.payoff) in [PayoffPut]:
            price = -self.s * m.exp(-self.y * self.expiry) * norm.cdf(-self.d1()) + self.payoff.strike * m.exp(
                -self.r * self.expiry) * norm.cdf(-self.d2())
        return price

    def delta(self):
        if type(self.payoff) in [PayoffCall]:
            delta = norm.cdf(self.d1())
        elif type(self.payoff) in [PayoffPut]:
            delta = norm.cdf(self.d1()) - 1
        return delta

    def gamma(self):
        return m.exp(-self.y * self.expiry) * norm.pdf(self.d1()) / self.s / self.sigma / m.sqrt(self.expiry)

    def gamma_strike(self):
        return -self.s / self.payoff.strike * self.gamma()

    def vega(self):
        return self.s * m.exp(-self.y * self.expiry) * norm.pdf(self.d1()) * m.sqrt(self.expiry)

    def strike_from_delta(self, delta_match, precision=10, n_iter_max=100):
        n_iter = 0
        option = copy.deepcopy(self)

        f = option.delta() - delta_match
        error = abs(f)

        while error > 10 ** (-precision) and n_iter < n_iter_max:
            f_prime = option.gamma_strike()
            # print f_prime
            # print -self.s / self.payoff.strike * self.gamma()
            option.set_strike(option.payoff.strike - f / f_prime)

            f = option.delta() - delta_match
            n_iter += 1
            error = abs(f)

        if n_iter == n_iter_max:
            print('Warning: raised n_iter_max in strike_from_delta method')

        output = option.payoff.strike
        del option
        return output

    def set_s(self, s):
        self.s = s
        self.sigma_update()

    def set_strike(self, strike):
        self.payoff.strike = strike
        self.sigma_update()

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_sigma_surface(self, sigma_surface):
        self.sigma_surface = sigma_surface
        self.sigma_update()


def implied_sigma(S, expiry, payoff, strike, r, price, eps=0.0001, sigma_min = 0.0, sigma_max = 1.0, max_iter=100):
    sigma = (sigma_min + sigma_max) / 2.0
    model = ModelBlackScholes(S, expiry, payoff(strike), r=r, sigma=sigma)
    model_price = model()
    error = 10000.0
    iter = 0

    while(error >= eps and iter <= max_iter):
        if model_price >= price:
            sigma_max = sigma
        else:
            sigma_min = sigma
        sigma = (sigma_min + sigma_max) / 2.0

        model.set_sigma(sigma)
        model_price = model()
        error = abs(model_price - price)
        iter += 1

    if iter > max_iter and error >= eps:
        return np.nan
    else:
        return sigma
