import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

symbols = ["EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDJPY", "USDCAD", "NZDUSD"]


all_symbols = dict()

for symbol in symbols:
    all_symbols[symbol] = pd.read_csv("myenvironment/{}_sync.csv".format(symbol))
    all_symbols[symbol].set_index("time", inplace=True)
    del all_symbols[symbol]["Unnamed: 0"]
    del all_symbols[symbol]["dumy"]


symbols_specs = {
    "EURUSD": {                
        "pip_value": 10,
        "volumen": 100000,
        "pip_decimal": 10000,
        "spread": 1*2
    } ,
    "GBPUSD": {                
        "pip_value": 10,
        "volumen": 100000,
        "pip_decimal": 10000,
        "spread": 1.3*2
    }  ,
    "NZDUSD": {                
        "pip_value": 10,
        "volumen": 100000,
        "pip_decimal": 10000,
        "spread": 1*2
    }  ,
    "AUDUSD": {                
        "pip_value": 10,
        "volumen": 100000,
        "pip_decimal": 10000,
        "spread": 1.1*2
    }  ,
    "USDCHF": {
        "volumen": 100000,
        "pip_decimal": 10000,
        "spread": 1.5*2
    },
    "USDCAD": {
        "volumen": 100000,
        "pip_decimal": 10000,
        "spread": 1.4*2
    },
    "USDJPY": {
        "volumen": 100000,
        "pip_decimal": 100,
        "spread": 1.3*2
    },

}


volumenByPos = {
        0: 0.01, # sell 0.01
        1: 0.01, # buy 0.01
        2: 0.01, # sell
        3: 0.01, # buy
        4: 0.02, # sell
        5: 0.02, # buy
        6: 0.03, # sell
        7: 0.03, # buy
        8: 0.05, # sell
        9: 0.05, #buy
        }

class EmptyOrder():
    def __init__(self, symbol):
        self.is_active = False
        self.symbol = symbol

class Order():

    def __init__(self, symbol, volumen, order_type, simulator):

        # order_type 0 to sell 1 to buy
        self.symbol = symbol
        self.volumen = volumen
        self.type = order_type
        self.is_active = True
        self.profit = 0
        self.max_posible_profit = 0
        self.max_posible_loss = 0
        self.opened_step = simulator.current_point
        self.simulator = simulator

    def update_stats(self):
        if self.is_active:

            open_price = self.simulator.data['{}_close'.format(self.symbol)].iloc[self.opened_step]
            current_price = self.simulator.data['{}_close'.format(self.symbol)].iloc[self.simulator.current_point]
            if self.type == 0:
                pips = open_price-current_price
            else:
                pips = current_price-open_price

            pips-=(self.simulator.syms_specs[self.symbol]["spread"]/self.simulator.syms_specs[self.symbol]["pip_decimal"])

            if  "pip_value" in self.simulator.syms_specs[self.symbol]:
                pip_value = self.simulator.syms_specs[self.symbol]["pip_value"] * self.volumen
            else:
                # Pip Value = (One Pip / Exchange Rate) * Lot Size
                pip_value = ((1 / self.simulator.syms_specs[self.symbol]["pip_decimal"]) / current_price)
                pip_value = pip_value * self.simulator.syms_specs[self.symbol]["volumen"] * self.volumen
            self.profit = pip_value * pips * self.simulator.syms_specs[self.symbol]["pip_decimal"]
            pass

    def close(self):

        if self.is_active:
            self.update_stats()
            self.simulator.current_equitiy += self.profit
            self.is_active = False


class Terminal():


    def __init__(self, all_symbols, syms_specs, initial_balance=1000, start_point=0, max_steps=1000):

        self.syms_specs = syms_specs
        self.origin_data = all_symbols
        # Flat symbols and set column
        sym_list = []
        for sym in self.origin_data:
            sym_list.append(self.origin_data[sym])

        self.main_data_source = pd.concat(sym_list, axis=1)
        columns=[]
        for col in symbols:
            columns.append("{}_open".format(col))
            columns.append("{}_high".format(col))
            columns.append("{}_low".format(col))
            columns.append("{}_close".format(col))

        self.back_periods = 60
        self.back_periods_ord=60
        self.group_by = 15 # 7 minuts
        self.main_data_source.columns = columns
        self.columns = columns
        self.account_currency = 'USD'
        self.orders = []
        self.initial_equitiy = initial_balance
        self.current_equitiy = self.initial_equitiy
        self.start_point = start_point
        self.max_steps = max_steps
        self.current_point = self.start_point
        self.historic_state = np.zeros((self.max_steps+self.back_periods_ord, 10*len(symbols)))


        self.orders_stack = dict()
        
        for x in range(len(all_symbols)):
            self.orders_stack[x] = dict()
            for xx in range(10):
                self.orders_stack[x][xx] = EmptyOrder(symbols[x])

        self.reset(start_point)


    def closeAll(self):
        for order in self.orders:
            if isinstance(order, Order):
                order.close()

    def openOrder(self, symbol, volumen, order_type):
        newOrder = Order(symbol, volumen, order_type, self)
        self.orders.append(newOrder)
        return newOrder

    def closeOrder(self, symbol, index):
        pass

    def reset(self, start_point=0):
        
        self.current_equitiy = self.initial_equitiy


        self.start_point = start_point

        selected_columns = []
        for x in symbols:
            selected_columns.append(("{}_open".format(x), '<lambda_0>'))
            selected_columns.append(("{}_high".format(x), 'amax'))
            selected_columns.append(("{}_low".format(x), 'amin'))
            selected_columns.append(("{}_close".format(x), '<lambda_1>'))
        
        columns=[]
        for col in symbols:
            columns.append("{}_open".format(col))
            columns.append("{}_high".format(col))
            columns.append("{}_low".format(col))
            columns.append("{}_close".format(col))


        self.data = self.main_data_source.iloc[self.start_point-(self.back_periods*self.group_by):self.start_point+(self.max_steps*self.group_by)].copy()
        self.data = self.data.reset_index()
        self.data = self.data.groupby(self.data.index // (self.group_by)).agg([lambda x: x.iloc[0], np.min, np.max, lambda x: x.iloc[-1]])    
        self.data = self.data[selected_columns]

        self.data.columns = columns

        self.start_point = self.back_periods
        self.current_point = self.start_point
        
        self.orders = []
        self.orders_stack = dict()
        self.historic_state = np.zeros((self.max_steps+self.back_periods_ord, 10*len(symbols)))
        for x in range(len(all_symbols)):
            self.orders_stack[x] = dict()
            for xx in range(10):
                self.orders_stack[x][xx] = EmptyOrder(symbols[x])

    def get_live_equitiy(self):
        ret = self.current_equitiy + sum([order.profit for order in self.orders if order.is_active])
        return ret

    def nextStep(self):
        if self.current_point + 1 >= self.start_point+self.max_steps:
            self.closeAll()
            return True

        self.current_point += 1
        for order in self.orders:
            if isinstance(order, Order):
                order.update_stats()

        return False

    def checkOrder(self, order, op, opidx):
        if order:
            if op == 0 and order.is_active:
                # close that order
                order.close()
                return order
            elif op == 1 and order.is_active:
                # nothing to do
                return order
            elif op == 1 and not order.is_active:
                #should open a order
                order_type = opidx % 2  
                volumen = volumenByPos[opidx]
                return self.openOrder(order.symbol, volumen, order_type)
            elif op == 0 and not order.is_active:
                # nothind to do
                return order


    def goAction(self, actionList):
        
        for idx, action in enumerate(actionList):
            binary_ops = '{0:08b}'.format(action)
            for opidx, op in enumerate(binary_ops):
                self.orders_stack[idx][opidx] = self.checkOrder(self.orders_stack[idx][opidx], int(op), opidx)


    def getCurrentState(self):
        vals_orig = self.data.iloc[self.current_point-self.back_periods: self.current_point].copy().values
        scaler = MinMaxScaler()
        vals = scaler.fit_transform(vals_orig)
        
        order_state = np.zeros((10*len(symbols)))
        idx = 0
        for key in self.orders_stack:
            for order in self.orders_stack[key]:
                if self.orders_stack[key][order] and self.orders_stack[key][order].is_active:
                    order_state[idx] = 1 
                idx+=1

        self.historic_state[self.current_point-1] = order_state
        
        # return vals.reshape(-1)
        return np.concatenate((vals.reshape(-1), self.historic_state[self.current_point-self.back_periods_ord:self.current_point,].reshape(-1 )))
        # return np.concatenate((vals.reshape(-1), order_state.reshape(-1 )))


class Environment():

    def __init__(self, min_range, max_range):
        self.simulator = Terminal(all_symbols, symbols_specs, initial_balance=1000, start_point=500000, max_steps=60)
        self.current_reward = 0
        self.current_equitiy = self.simulator.get_live_equitiy()
        self.min_range = min_range
        self.max_range = max_range

    def observation_space(self):
        return (self.simulator.getCurrentState().shape[0],)

    def action_space(self):
        return len(symbols)

    def reset(self):
        new_point = np.random.randint(self.min_range, len(self.simulator.main_data_source)-self.max_range)
        self.simulator.reset(new_point)
        self.current_equitiy = self.simulator.get_live_equitiy()
        return self.simulator.getCurrentState()

    def render(self, close=False):
        print("Current EQ", self.current_equitiy)
        print("Start Point: ", self.simulator.start_point, " end porint: ", self.simulator.current_point, " steps_pending: ",  self.simulator.current_point-self.simulator.start_point)
        print("Orders Active: ")
        for key in self.simulator.orders_stack:
            for order in self.simulator.orders_stack[key]:
                if self.simulator.orders_stack[key][order] and self.simulator.orders_stack[key][order].is_active:
                    print(self.simulator.orders_stack[key][order].symbol, self.simulator.orders_stack[key][order].volumen, self.simulator.orders_stack[key][order].type)

    def step(self, action_list):
        
        self.simulator.goAction(action_list)
        done = self.simulator.nextStep()

        # if not done:
        newEquitiy = self.simulator.get_live_equitiy()
        reward = newEquitiy - self.current_equitiy
        self.current_equitiy = newEquitiy
        # else:
        # reward = self.simulator.get_live_equitiy()
        

        new_state = self.simulator.getCurrentState()
        return new_state, reward, done, None
        #return newstate, reward, done



def main():
    env = Environment(500_000, 500_000)
    print(env.observation_space())
    print(env.action_space())
    new_state, reward, done, _ = env.step([255, 255,1, 45, 45, 45, 1]) 
    print(reward)
    new_state, reward, done, _ = env.step([2, 255,1, 45, 40, 45, 1]) 
    print(reward)
    new_state, reward, done, _ = env.step([3, 255,1, 45, 40, 45, 1]) 
    print(reward)
    new_state, reward, done, _ = env.step([3, 0,255, 45, 255, 3, 1]) 
    print(reward)
    new_state, reward, done, _ = env.step([2, 255,1, 45, 40, 45, 1]) 
    print(reward)
    print(new_state)
if __name__ == '__main__':
    main()
