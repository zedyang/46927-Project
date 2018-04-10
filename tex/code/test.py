# Scope: Simulator(object):

    def single_path(self):
        # initialize market
        spot_mkt = self.spot_process.sample()
        options_mkt = dict()
        if self.rates_type == 'Fixed':
            rates_mkt = [self.rates] * (self.n_period+1)
        
        time_stamps = [self.t0 + j*self.dt for j in range(self.n_period+1)]
        if self.ts_format is float:
            for option in self.options:
                options_mkt[option.code] = [EuropeanOption.bsm_price(
                    t=time_stamps[j], S=spot_mkt[j], 
                    K=option.strike, T=option.expiry_date, 
                    sigma=self.implied_vol, 
                    r=rates_mkt[j],
                    q=self.carry)[option.type] 
                    for j in range(self.n_period+1)
                ]
        elif self.ts_format is datetime:
            for option in self.options:
                options_mkt[option.code] = [EuropeanOption.bsm_price(
                    t=0, S=spot_mkt[j], 
                    K=option.strike, 
                    T=option.time_to_expiry(
                        time_stamps[j], self.calendar), 
                    sigma=self.implied_vol, 
                    r=rates_mkt[j],
                    q=self.carry)[option.type] 
                    for j in range(self.n_period+1)
                ]
        # initialize portfolio
        quote_0 = MarketQuotes(
            spot=spot_mkt[0],
            options=dict([(option.code, 
                           options_mkt[option.code][0]) 
                          for option in self.options]),
            rate=rates_mkt[0],
            time=time_stamps[0]
        )
        self.pfolio.update_quotes(quote_0)
        for opt, pos in self.position_size.iteritems():
            self.pfolio.long_short(key=opt, val=pos)
        delta_0 = self.pfolio.delta_with_vol(self.hedging_vol)
        self.pfolio.long_short(val=-delta_0)
        init_haircut = np.abs(self.pfolio.value) / self.leverage
        # main loop
        for j in range(1, self.n_period+1):
            quote = MarketQuotes(
                spot=spot_mkt[j],
                options=dict([(option.code, 
                               options_mkt[option.code][j])
                              for option in self.options]),
                rate=rates_mkt[j],
                time=time_stamps[j]
            )
            self.pfolio.update_quotes(quote)
            self.hedger.rehedge(quote, self.hedging_vol)
        return self.pfolio.pnl, init_haircut