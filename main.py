from QuantConnect import *
from QuantConnect.Parameters import *
from QuantConnect.Benchmarks import *
from QuantConnect.Brokerages import *
from QuantConnect.Util import *
from QuantConnect.Interfaces import *
from QuantConnect.Algorithm import *
from QuantConnect.Algorithm.Framework import *
from QuantConnect.Algorithm.Framework.Selection import *
from QuantConnect.Algorithm.Framework.Alphas import *
from QuantConnect.Algorithm.Framework.Portfolio import *
from QuantConnect.Algorithm.Framework.Execution import *
from QuantConnect.Algorithm.Framework.Risk import *
from QuantConnect.Indicators import *
from QuantConnect.Data import *
from QuantConnect.Data.Consolidators import *
from QuantConnect.Data.Custom import *
from QuantConnect.Data.Fundamental import *
from QuantConnect.Data.Market import *
from QuantConnect.Data.UniverseSelection import *
from QuantConnect.Notifications import *
from QuantConnect.Orders import *
from QuantConnect.Orders.Fees import *
from QuantConnect.Orders.Fills import *
from QuantConnect.Orders.Slippage import *
from QuantConnect.Scheduling import *
from QuantConnect.Securities import *
from QuantConnect.Securities.Equity import *
from QuantConnect.Securities.Forex import *
from QuantConnect.Securities.Interfaces import *
from datetime import date, datetime, timedelta
from QuantConnect.Python import *
from QuantConnect.Storage import *
QCAlgorithmFramework = QCAlgorithm
QCAlgorithmFrameworkBridge = QCAlgorithm

class CoIntegratedPairsTrading(QCAlgorithm):

    def Initialize(self):
        ## SET TICKERS 
        tickers_energy=["XLE","IYE","VDE","USO","XES","XOP","UNG","ICLN","ERX","ERY","SCO","UCO","AMJ","BNO","AMLP","OIH","DGAZ","UGAZ","TAN"]
        tickers_metals=["GLD", "IAU", "SLV", "GDX", "AGQ","GDXJ", "PPLT", "NUGT", "USLV", "UGLD", "JNUG", "DUST", "JDST"]
        #tickers_treasuries=["IEF", "SHY", "TLT","SHV", "IEI", "TLH", "EDV", "BIL", "SPTL","VGSH","VGIT","VGLT", "TMF", "SCHO", "SCHR", "SPTS", "GOVT","SHV", "TBT", "TMV"]
        tickers_treasuries=["IEF", "SHY", "TLT","IEI", "TLH", "EDV", "BIL", "SPTL","VGSH","VGIT","VGLT",
                "TMF", "SCHO", "SCHR", "SPTS", "GOVT","SHV", "TBT", "TMV"]
        tickers_tech=["XLK","QQQ","SOXX","IGV","VGT","QTEC","FDN","FXL","TECL","TECS","SOXL","SOXS","SKYY","SMH","KWEB","FTEC"]
        tickers_vol=["SQQQ","TQQQ","TVIX","VIXY","SPLV","SVXY","UVXY","EEMV","EFAV","USMV"]

        tickers_all= [*tickers_energy, *tickers_metals,*tickers_treasuries,*tickers_tech,*tickers_vol]
        
        ## SET PARAMETERS
        self.SetStartDate(2018, 6, 30)  # Set Start Date
        self.SetEndDate(2019, 1, 27) # Set End Date
        self.SetCash(100000)  # Set Strategy Cash
        self.SetBenchmark("SPY") # Set Benchmark
        resolution = Resolution.Minute # Set Resolution
        # self.UniverseSettings.Resolution = Resolution.Hour # Set resolution

        self.symbols = {}

        price_all = pd.DataFrame(columns=tickers_all)

        for i in range(len(self.tickers_all)):
            self.symbols[self.tickers_all[i]] = self.AddEquity(self.tickers_all[i], resolution).Symbol
            price_all[price_all.columns[i]] = self.History([tickers_all[i]],start ,end, Resolution.Daily).loc[tickers_all[i]]['close']
        price_all.index = price_all.index.date

            

# self.Schedule.On(self.DateRules.EveryDay("VTX"), self.TimeRules.AfterMarketOpen("VTX", 10), Action(self.))

    def signals(self):
        pass
    

    def olsRegression(self, tickers):
        for i in tickers:
            x = np.log(data[data.columns[tickers.index(i)]])
            x_const = sm.add_constant(x)
            tickers2=[t for t in tickers if t !=i]
            for j in tickers2:
                y = np.log(data[data.columns[tickers.index(j)]])
                linear_reg = sm.OLS(y,x_const)
                results = linear_reg.fit()
                results_dict[(i,j)]=[]
                results_dict[(i,j)].append(results.params[0])
                results_dict[(i,j)].append(results.params[1])
        # Save dict to dataframe
        results_df=pd.DataFrame.from_dict(results_dict)
        results_df.rename(index={0: "alpha", 1:'beta'})

        x_var=[]
        y_var=[]
        for i in range(len(results_df.columns)):
            x_var.append(results_df.columns[i][0])
            y_var.append(results_df.columns[i][1])

        y_predicted={}
        for i in range(len(results_df.columns)):
            y_predicted[(x_var[i],y_var[i])]=[]
            y_predicted[(x_var[i],y_var[i])].append(np.log(data[x_var[i]])*results_df.loc[1][i]+results_df.loc[0][i])
        
        #Save predicted y-values in a dataframe
        # mycols=list(y_predicted.keys())
        # y_df=pd.DataFrame(columns=mycols)
        # for i in range(len(y_predicted.values())):
        #     colname=mycols[i]
        #     y_df.loc[:,colname]=pd.DataFrame(list(y_predicted.values())[i]).transpose().iloc[:,0]

        # construct the spread series based on the OLS estimates
        mycols=(list(y_predicted.keys()))
        spread=pd.DataFrame(columns=mycols)

        # construct the spread series based on the OLS estimates
        for i in range(len(mycols)):
            colname=mycols[i]
            beta=results_df.loc[1][i]
            alpha=results_df.loc[0][i]
            spread.loc[:,colname]=pd.DataFrame(np.log(data[y_var[i]])-np.log(data[x_var[i]])*beta-alpha).iloc[:,0]

        #Use pythons stattools pack to run the ADF test and check the stationarity of spreads
        #print relevant stats that we'll use later to filter highly co-integrated pairs. 
        mycols_adf=spread.columns
        adf_df=pd.DataFrame(columns=mycols_adf)
        index=["ADF test statistics","ADF 1%", "ADF 5%", "ADF 10%", "p-value"]
        for i in range(len(spread.columns)):
            col=mycols_adf[i]
            adf=sm.tsa.stattools.adfuller(spread[spread.columns[i]], maxlag=1)
            adf_df.loc[index[0],col] = adf[0]
            adf_df.loc[index[1],col] = list(adf[4].values())[0]
            adf_df.loc[index[2],col] = list(adf[4].values())[1]
            adf_df.loc[index[3],col] = list(adf[4].values())[2]
            adf_df.loc[index[4],col] = adf[1]

        #Filter the results where we reject the null hypothesis with a significance level of less than 1%.
        #in practical term this means that ADF statistics should be less than ADF 1% and/or
        #p-value <0.01

        filtered_pairs=list(adf_df.columns[adf_df.loc[adf_df.index[4]]<0.01])
        filtered_spread=spread[filtered_pairs]

        cols3=filtered_spread.columns
        trading_signals=pd.DataFrame(columns=cols3)
        index2=["mean spread","upper threshold", "lower_threshold"]
        stdev=1.00 #We can impose a more restrictive condition (e.g. 1.96 sd if we want to trade less frequently)
        for i in range(len(cols3)):
            trading_signals.loc[index2[0],cols3[i]] = filtered_spread[filtered_spread.columns[i]].mean()
            trading_signals.loc[index2[1],cols3[i]] = filtered_spread[filtered_spread.columns[i]].mean()+stdev*filtered_spread[filtered_spread.columns[i]].std()
            trading_signals.loc[index2[2],cols3[i]] = filtered_spread[filtered_spread.columns[i]].mean()-stdev*filtered_spread[filtered_spread.columns[i]].std()

        #Here buying the spread means buy 1 unit of independent var(y) and sell
        #beta unit of the dependent variable(x).
        #We expect that the relationship between x and y will hold in the future. 
        #Buying the spread when it is lower than the standard deviation and closing 
        #out the position when it returns to mean. 
        #Selling the spread means to sell 1 unit of  Y and 
        #buy beta units of x when it is above sigma,
        #and close the position when reaching the long-term mean to realize a profit.
        df_buy=pd.DataFrame(columns=cols3) 
        for i in range(len(filtered_spread.columns)):
            c=filtered_spread[filtered_spread.columns[i]]
            df_buy.loc[:,cols3[i]]=((c < trading_signals.iloc[2,i]) & (c.shift(1) > trading_signals.iloc[2,i]) | 
                          (c <  trading_signals.iloc[0,i]) & (c.shift(1) >  trading_signals.iloc[0,i]))
        
        df_sell=pd.DataFrame(columns=cols3) 
        for i in range(len(filtered_spread.columns)):
            c=filtered_spread[filtered_spread.columns[i]]
            df_sell.loc[:,cols3[i]]=((c > trading_signals.iloc[1,i]) & (c.shift(1) > trading_signals.iloc[1,i]) | 
                          (c >  trading_signals.iloc[0,i]) & (c.shift(1) <  trading_signals.iloc[0,i]))

        #Create dataframe with trading signals based on Boolean dataframe above
        #buying the spread implied-> buy 1 unit of y-variable (second ticker), 
        #sell beta units of x-variable (first ticker)
        buyspreadtemp=pd.DataFrame(columns=cols3, index=df_buy.index) 
        for i in range(len(df_buy.columns)):
            for j in range(len(df_buy.index)):
                if df_buy.iloc[j,i]==False:
                    buyspreadtemp.iloc[j,i]= [0,0]
                else: 
                    buyspreadtemp.iloc[j,i]= [-round(results_df.iloc[1,i],0),1] #negative sign denotes sell position

        #Use the temporary dataframe to create the final dataframe of buy spread trading signals
        for j in range(1,len(df_buy.index)):
            for i in range(len(df_buy.columns)):
                if buyspread.iloc[j,i]==[0,0]:
                    buyspread.iloc[j,i]=buyspread.iloc[j-1,i]

        # create datetime index in the returns_data series
        returns_data.index=pd.to_datetime(returns_data.index)

        buyspreadreturns=pd.DataFrame(columns=buyspread.columns, index=buyspread.index.copy())

        # Add returns data for filtered buy spreads
        for j in buyspreadreturns.index:
            for i in range(len(buyspreadreturns.columns)):
                firstticker=buyspreadreturns.columns[i][0]
                secondticker=buyspreadreturns.columns[i][1]
                dateindex=buyspreadreturns.index.get_loc(j)
                #dt_ind = j.date()
                element1=round(returns_data.loc[j,firstticker],3)
                element2=round(returns_data.loc[j,secondticker],3)
                buyspreadreturns.iloc[dateindex,i]=[element1, element2] 

        pair_ret_buy_temp=buyspread.copy()
        for i in range(len(pair_ret_buy_temp.columns)):
            for j in range(len(pair_ret_buy_temp.index)):
                pair_ret_buy_temp.iloc[j,i]=np.array(buyspread.iloc[j][i])*np.array(buyspreadreturns.iloc[j][i])
        #pair_ret_buy_temp.head()

        # Estimate returns for each pair trading on a given day
        pair_ret_buy=pair_ret_buy_temp.copy()
        for i in range(len(pair_ret_buy_temp.columns)):
            for j in range(len(pair_ret_buy_temp.index)):
                pair_ret_buy.iloc[j,i]=sum(pair_ret_buy_temp.iloc[j][i])

        #Select the pair with the highest cumulative return for each ticker
        pair_ret_buy_cum=pair_ret_buy.cumsum()
        cumval_buy=pair_ret_buy_cum.iloc[[-1]] #cumulative returns for all filtered pairs

        buyindex=[]
        for i in range(len(pair_ret_buy.columns)):
            buyindex.append(pair_ret_buy.columns[i][0])
        buyindex_unique=list(set(buyindex))

        def getIndexes(dfObj, value):
            ''' Get index positions of value in dataframe i.e. dfObj.'''
            listOfPos = list()
            # Get bool dataframe with True at positions where the given value exists
            result = dfObj.isin([value])
            # Get list of columns that contains the value
            seriesObj = result.any()
            columnNames = list(seriesObj[seriesObj == True].index)
            # Iterate over list of columns and fetch the rows indexes where value exists
            for col in columnNames:
                rows = list(result[col][result[col] == True].index)
                for row in rows:
                    listOfPos.append((row, col))
            # Return a list of tuples indicating the positions of value in the dataframe
            return listOfPos

        selectedpairs_buyspread=[]
        for i in range(len(buyindex_unique)):
            df2 = cumval_buy.filter(regex=buyindex_unique[i])
            maxval= df2.max().max()
            selected_pair= getIndexes(df2, maxval)[0][1]
            selectedpairs_buyspread.append(selected_pair)
        selectedpairs_buyspread=list(set(selectedpairs_buyspread))

        #Create dataframe with trading signals based on Boolean dataframe above.
        #Selling the spread implies-> #buy beta units of x-variable
        #sell 1 unit of y-variable, 
        sellspreadtemp=pd.DataFrame(columns=cols3, index=df_sell.index) 
        for i in range(len(df_sell.columns)):
            for j in range(len(df_sell.index)):
                if df_sell.iloc[j,i]==False:
                    sellspreadtemp.iloc[j,i]= [0,0]
                else: 
                    sellspreadtemp.iloc[j,i]= [round(results_df.iloc[1,i],0),-1]

        # Add returns data for filtered sell spreads
        sellspreadreturns=pd.DataFrame(columns=sellspread.columns, index=sellspread.index.copy())

        # Add returns data for filtered sell spreads
        for j in sellspreadreturns.index:
            for i in range(len(sellspreadreturns.columns)):
                firstticker=sellspreadreturns.columns[i][0]
                secondticker=sellspreadreturns.columns[i][1]
                dateindex=sellspreadreturns.index.get_loc(j)
                #dt_ind = j.date()
                element1=round(returns_data.loc[j,firstticker],3)
                element2=round(returns_data.loc[j,secondticker],3)
                sellspreadreturns.iloc[dateindex,i]=[element1, element2]

        pair_ret_sell_temp=sellspread.copy()
        for i in range(len(pair_ret_sell_temp.columns)):
            for j in range(len(pair_ret_sell_temp.index)):
                pair_ret_sell_temp.iloc[j,i]=np.array(sellspread.iloc[j][i])*np.array(sellspreadreturns.iloc[j][i])

        # Estimate returns for each pair trading on a given day
        pair_ret_sell=pair_ret_sell_temp.copy()
        for i in range(len(pair_ret_sell_temp.columns)):
            for j in range(len(pair_ret_sell_temp.index)):
                pair_ret_sell.iloc[j,i]=sum(pair_ret_sell_temp.iloc[j][i])

        #Select the pair with the highest cumulative return for each ticker
        pair_ret_sell_cum=pair_ret_sell.cumsum()
        cumval_sell=pair_ret_sell_cum.iloc[[-1]] #cumulative returns for all filtered pairs

        sellindex=[]
        for i in range(len(pair_ret_sell.columns)):
            sellindex.append(pair_ret_sell.columns[i][0])
        sellindex_unique=list(set(sellindex))

        selectedpairs_sellspread=[]
        for i in range(len(sellindex_unique)):
            df2 = cumval_sell.filter(regex=sellindex_unique[i])
            maxval= df2.max().max()
            selected_pair= getIndexes(df2, maxval)[0][1]
            selectedpairs_sellspread.append(selected_pair)
        selectedpairs_sellspread=list(set(selectedpairs_sellspread))

        for i in range(len(commonelements)):
            el= commonelements[i]
            if np.array(cumval_sell[el]>cumval_buy[el])==True:
                selectedpairs_buyspread.remove(el)
            else: 
                selectedpairs_sellspread.remove(el)

        return selectedpairs_buyspread, selectedpairs_sellspread


    # totalselectedpairs=selectedpairs_buyspread+selectedpairs_sellspread
    # # totalselectedpairs
    # #Assuming an equal portfolio allocation for each pair we can assing the following weights:
    # weight_buyspread=len(selectedpairs_buyspread)/len(totalselectedpairs)
    # weight_sellspread=len(selectedpairs_sellspread)/len(totalselectedpairs)
    # weight_buyspread+weight_sellspread
        
    def momentum_indicator(self, price_all):
        ema_short = price_all.ewm(span=20, adjust=False).mean()
        ema_long = price_all.ewm(span=50, adjust=False).mean()

        trading_positions_raw = price_all - ema_short
        #Momentum indicator summary genereated from price vs. short EMA comparison
        momindicator=trading_positions_raw.apply(np.sign)
        return momindicator

    def pairsTrading(self):
        #Find average momentum indicator for each symbol and check its consistency
        #with the signal obtained from buy-spread pairs trading strategy 
        final_list_pairs_buyspread=[]
        for i in range(len(selectedpairs_buyspread)):
            pair=selectedpairs_buyspread[i]
            symbol_1=selectedpairs_buyspread[i][0]
            symbol_2= selectedpairs_buyspread[i][1]
    #     Estimate average sentiment for each symbol in our pair
            avgmomfreq1=momindicator[symbol_1].mean()
            avgmomfreq2=momindicator[symbol_2].mean()
            if (avgmomfreq1<0 and avgmomfreq2>0):
                final_list_pairs_buyspread.append(pair)

        #Find average momentum indicator for each symbol and check its consistency
        #with the signal obtained from sell-spread pairs trading strategy 
        final_list_pairs_sellspread=[]
        for i in range(len(selectedpairs_sellspread)):
            pair=selectedpairs_sellspread[i]
            symbol_1=selectedpairs_sellspread[i][0]
            symbol_2= selectedpairs_sellspread[i][1]
        #     Estimate average sentiment for each symbol in our pair
            avgmomfreq1=momindicator[symbol_1].mean()
            avgmomfreq2=momindicator[symbol_2].mean()
            if (avgmomfreq1>0 and avgmomfreq2<0):
                final_list_pairs_sellspread.append(pair)
            
        final_list_pairstrading=final_list_pairs_buyspread+final_list_pairs_sellspread

        #Trading positions for our final list of insights
        buyspread[final_list_pairs_buyspread]
        sellspread[final_list_pairs_sellspread]
        pd.concat([buyspread[final_list_pairs_buyspread], sellspread[final_list_pairs_sellspread]],axis=1)


    def rebalance(self):
        pairstrading(self)

    def common_elements(list1, list2):
        return [element for element in list1 if element in list2]

    def OnData(self, data):
        pass
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''

        # if not self.Portfolio.Invested:
        #    self.SetHoldings("SPY", 1)