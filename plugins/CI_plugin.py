import os
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from pygcam.subcommand import SubcommandABC
from pygcam.config import getParam
from pygcam.log import getLogger
from pygcam.error import PygcamException
from pygcam.query import readCsv, writeCsv
from pygcam.sectorEditors import (
    TECH_CELLULOSIC_ETHANOL, TECH_SUGARCANE_ETHANOL, TECH_CORN_ETHANOL,
    TECH_FT_BIOFUELS, TECH_BIODIESEL, TECH_CTL, TECH_GTL)
from pygcam.units import getUnits

_logger = getLogger(__name__)

RegionsUSA = ['USA', 'United States']

LAND_USE_CHANGE = 'Land-use Change'

ETOH = 'EtOH'
FAME = 'FAME'
CORN_BIOFUELS = 'corn ethanol'

CO2_PER_C  = 44.0/12.0

# AR5 GWPs include climate-carbon feedback
CH4_GWP_100 = 34
N2O_GWP_100 = 298
CO2_GWP_100 = 1

# Double checked these against mappings/emissions.csv on 6/8/18. All good.
GcamCO2eFactors = {
    'BC':           0,
    'CO':           0,
    'CO2':          CO2_PER_C,      # in GCAM, "CO2" is in units of Tg C, so this is not a GWP, but the C->CO2 factor
    'CH4':          CH4_GWP_100,
    'N2O':          N2O_GWP_100,
    'NH3':          0,
    'NMVOC':        0,
    'NOx':          0,
    'OC':           0,
    'SF6':          22800 / 1000.0, # high-GWP gases are mislabeled Tg in GCAM; actually in Gg so we divide by 1000
    'Sulfur':       0,
    'C2F6':     12200 / 1000.0, # GWP in numerator, conversion of Gg to Mg in denominator (checked)
    'CF4':       7390 / 1000.0, # (checked)
    'HFC23':    14800 / 1000.0, # (checked)
    'HFC32':      675 / 1000.0, # (checked)
    'HFC43':     1640 / 1000.0, # (checked)
    'HFC125':    3500 / 1000.0, # (checked)
    'HFC134a':   1430 / 1000.0, # (checked)
    'HFC143a':   4470 / 1000.0, # (checked)
    'HFC152a':    124 / 1000.0, # (checked)
    'HFC236fa':  9810 / 1000.0, # (checked)
    'HFC245fa':  1030 / 1000.0, # (checked)
    'HFC227ea':  3220 / 1000.0, # (checked)
    'HFC365mfc':  794 / 1000.0, # (checked)
}

# The queries used herein
Q_REFINED_LIQUIDS = 'refined_liquids_production_by_tech'
Q_LUC_EMISSIONS   = 'luc_emissions'
Q_GHG_EMISSIONS   = 'nonco2'        # query name is a misnomer: results include CO2

QueryList = [Q_REFINED_LIQUIDS, Q_LUC_EMISSIONS, Q_GHG_EMISSIONS]

# EJ of biomass oil required per EJ of biodiesel
# BiodieselOilCoefficients = [(2015, 1.039), (2020, 1.026), (2025, 1.012), (2030, 1.008)
#                             (2035, 1.003), (2040, 0.999), (2045, 0.994), (2050, 0.989)]

# TBD: decide whether to keep this
BioOilToBiodieselCoef = 1.03


class CarbonIntensityCommand(SubcommandABC):
    def __init__(self, subparsers):
        kwargs = {'help' : '''Plugin to compute carbon intensity values for the given baseline and
            policy scenarios and save the result to the given output CSV or XLSX file.'''}

        super(CarbonIntensityCommand, self).__init__('CI', subparsers, kwargs)

    def addArgs(self, parser):
        parser.add_argument('-b', '--baseline', required=True,
                            help='''The name of the baseline scenario''')

        parser.add_argument('-d', '--diffsDir', default='.',
                            help='''Specify the directory with the '.csv' files with the differences between
                            scenarios, which must include the following query results: %s''' % \
                                 [name + '.csv' for name in QueryList])

        parser.add_argument('-f', '--fuelShockFile', default=None,
                            help='''The name of the file to write out the fuel shock size in EJ.
                                    This can be used by scripts that require unit conversions, e.g., per MJ
                                    of fuel. Default is no output file''')

        parser.add_argument('-p', '--scenario', dest='policy', metavar='scenario', required=True,
                            help='''The name of the policy scenario''')

        parser.add_argument('-y', '--years', type=str, default="2020-2050",
                            help='''Takes a parameter of the form XXXX-YYYY, indicating start and end
                            years of interest. Other years are dropped.''')

        parser.add_argument('-Y', '--shockStartYear', type=int, default=None,
                            help='''Specify the first year for which to count changes in the target fuel.
                                If not specified, defaults to the value given in the -y/--years argument.''')

        return parser

    def run(self, args, tool):
        # If not passed on command-line, read from config file
        yearsStr = args.years or getParam('GCAM.Years')
        years = [int(s) for s in yearsStr.split('-')]

        if len(years) != 2:
            raise PygcamException('''Years must be specified as XXXX-YYYY, where XXXX and YYYY
                                     are the first and last years to consider, respectively''')

        firstYear, lastYear = years

        guess = shockedFuel(args.policy)
        _logger.info("Inferred shocked fuel '%s' from policy '%s'", guess, args.policy)
        shocked = [guess]

        GcamResultProcessor(args.baseline, args.policy, args.diffsDir, shocked,
                            args.fuelShockFile, firstYear=firstYear, lastYear=lastYear)


PluginClass = CarbonIntensityCommand


def log_fmt_df(df):
    _logger.info('\n' + df.to_string(formatters={'value': '{:,.2f}'.format}) + '\n')


def yearColumns(df):
    return [c for c in df.columns if c.isdigit()]


class GcamResultProcessor(object):
    def __init__(self, baseline, policy, diffsDir, shocked, fuelShockFile,
                 firstYear=2020, lastYear=2050):
        self.baseline = baseline
        self.policy = policy
        self.firstYear = firstYear
        self.lastYear = lastYear
        self.diffsDir = diffsDir
        self.queryList = QueryList
        self.yearsOfInterest = range(firstYear, lastYear + 1)
        self.fuelShockFile = fuelShockFile
        self.fuelEJ = None

        self.verifyResultFiles(policy)  # raises error if anything is missing

        # Create empty DF to store annual data. Data and additional columns will be added later.
        self.GHG = pd.DataFrame(index=[str(y) for y in range(firstYear, lastYear)], columns=['CO2e'], data=0)

        # Read all difference files associated with the queries in QueryList
        self.diffDFs = self.readDiffs()

        # Modify a copy of the LUC emissions, converted to CO2e
        lucCO2 = self.diffDFs[Q_LUC_EMISSIONS]
        yearCols = yearColumns(lucCO2)

        lucCO2 = lucCO2.rename(columns={'land-use-change-emission': 'sector'}, inplace=False)
        lucCO2[yearCols] *= CO2_PER_C
        lucCO2['sector'] = LAND_USE_CHANGE
        lucCO2['Units'] = 'Mt CO2'

        self.lucCO2 = lucCO2

        self.normedDF = self.save_normalized_GHGs()
        self.save_gas_categories()
        self.calc_carbon_intensity()
        self.save_luc_emissions()

    def queryPathname(self, queryName):
        pathname = os.path.join(self.diffsDir, '{}-{}-{}.csv'.format(queryName, self.policy, self.baseline))
        return pathname

    def verifyResultFiles(self, scenario):
        def resultFileDoesntExist(queryName):
            filename = self.queryPathname(queryName)
            path = os.path.join(self.diffsDir, filename)
            _logger.debug("Checking for '%s'", path)
            return (None if os.path.lexists(path) else path)

        # find missing files, if any
        names = list(filter(resultFileDoesntExist, self.queryList))
        if names:
            raise PygcamException("Query result files are missing in %s for %s scenario:\n  %s" % \
                                    (self.diffsDir, scenario, "\n  ".join(names)))

    def readDiff(self, query):
        '''
        Read a single query and return a DF holding the results.
        '''
        path = self.queryPathname(query)
        df = readCsv(path)
        return df

    def readDiffs(self):
        '''
        Read the given list of queries results for the given scenario into DFs.
        Return a dict keyed by query name, with the corresponding DF as the value.
        '''
        _logger.debug("Loading results")

        results = {q: self.readDiff(q) for q in self.queryList}

        # 0.0 indicates a non-existent category created by the label re-write
        results[Q_GHG_EMISSIONS] = results[Q_GHG_EMISSIONS].query('Units != 0').copy()
        return results

    def filterValues(self, df, column, values, complement=False):
        '''
        Restore the columns as such, perform the query, isolate the
        rows of interest, and set the index back to the original value.
        Return a new df with only rows matching the filter criteria.
        '''
        if not hasattr(values, '__iter__'):
            values = [values]

        query = "{} {} in {}".format(column, 'not' if complement else '', values)
        df = df.query(query)
        return df

    def sumDiffs(self, df, col=None, values=None, complement=False, startYear=None):
        '''
        Sum the year-by-year differences and then sum across years. If col and
        values are provided, only rows with (or without, if complement is True)
        one of the given values in the given column will be included in the sum. The
        parameter startYear allows fuel changes to be counted starting at the shock
        year rather than counting the values interpolated from the baseline to the
        value in the first shock year.
        '''
        if col and values:
            df = self.filterValues(df, col, values, complement=complement)

        firstYear = startYear if startYear is not None else self.firstYear
        years = range(firstYear, self.lastYear + 1)
        yearCols = [str(y) for y in years]
        yearTotals = df[yearCols].sum() # sum columns
        total = yearTotals.sum()        # sum across years
        return total

    def save_cumulative_GHGs(self):
        normedDF = self.normedDF

        yearCols = yearColumns(normedDF)
        cumLandUseDF = normedDF.query("sector == '{}'".format(LAND_USE_CHANGE))

        df = pd.DataFrame(columns=yearCols, index=['luc-only', 'all-ghgs'])
        df.loc['luc-only'] = cumLandUseDF[yearCols].sum()
        df.loc['all-ghgs'] = normedDF[yearCols].sum()
        df.index = df.index.rename('method')
        df.reset_index(inplace=True)

        cumEmissionsFile = self.queryPathname('cumulative_emissions')
        writeCsv(df, cumEmissionsFile, header='Cumulative Emissions', float_format="%.3f")

    def save_gas_categories(self):
        USA = 'United States'
        ROW = 'Rest of world'
        AG_N2O = 'Agriculture N2O'
        AG_CH4 = 'Agriculture CH4'
        LS_N2O = 'Livestock N2O'
        LS_CH4 = 'Livestock CH4'
        LUC_CO2 = 'Land-use Change CO2'

        nonCO2Diffs = self.diffDFs[Q_GHG_EMISSIONS]
        cropCH4 = nonCO2Diffs.query('sector == "CropProduction|Total" and GHG == "CH4"')
        cropN2O = nonCO2Diffs.query('sector == "CropProduction|Total" and GHG == "N2O"')

        livestockCH4 = nonCO2Diffs.query('sector == "LivestockProduction" and GHG == "CH4"')
        livestockN2O = nonCO2Diffs.query('sector == "LivestockProduction" and GHG == "N2O"')

        lucCO2 = self.lucCO2

        co2e = {USA: defaultdict(float), ROW: defaultdict(float)}

        regions = [USA, ROW]

        def sumDiffHelper(df, complement):
            # Reduces redundancy in the loop below.
            return self.sumDiffs(df, col='region', values=RegionsUSA, complement=complement)

        for region in regions:
            # Converts any regionalization into USA and not USA (i.e., ROW)
            complement = region not in RegionsUSA
            co2e[region][LUC_CO2] = sumDiffHelper(lucCO2,       complement)
            co2e[region][AG_CH4]  = sumDiffHelper(cropCH4,      complement) * CH4_GWP_100
            co2e[region][AG_N2O]  = sumDiffHelper(cropN2O,      complement) * N2O_GWP_100
            co2e[region][LS_CH4]  = sumDiffHelper(livestockCH4, complement) * CH4_GWP_100
            co2e[region][LS_N2O]  = sumDiffHelper(livestockN2O, complement) * N2O_GWP_100

        # turn it into a DataFrame to save as csv for plotting
        lastYearStr = str(self.lastYear)
        rowDicts = [[{'region': region, 'output': key, lastYearStr: value} \
                     for key, value in co2e[region].items()] \
                        for region in regions]
        df = pd.DataFrame(data=list(itertools.chain.from_iterable(rowDicts)))
        df['Units'] = 'Tg CO2e'

        # Save component results for plotting
        filename = self.queryPathname('Emissions-changes')
        writeCsv(df, filename, header='Emissions changes')

    def calc_carbon_intensity(self):
        diffs = self.diffDFs
        normedDF = self.normedDF

        # Sum just the LUC CO2 emissions
        lucCO2  = self.sumDiffs(normedDF.query("sector == '{}'".format(LAND_USE_CHANGE)))

        nonCO2Diffs = self.diffDFs[Q_GHG_EMISSIONS]

        kyotoCO2e = self.sumDiffs(normedDF)

        _logger.debug("    Total GHG: {:8.2f} Tg CO2e (AR5 GWPs)".format(kyotoCO2e))
        _logger.debug("      LUC CO2: {:8.2f} Tg CO2e".format(lucCO2))
        _logger.debug('')

        refinedLiquids = diffs[Q_REFINED_LIQUIDS]
        diffsUSA = refinedLiquids.query('region in {}'.format(RegionsUSA))
        diffsROW = refinedLiquids.query('region not in {}'.format(RegionsUSA))

        shocked = self.shocked[0] if len(self.shocked) else None

        if shocked:
            yearCols = yearColumns(refinedLiquids)

            # TBD: Review whether to keep this
            if shocked == TECH_CORN_ETHANOL:
                df = diffsUSA.query('output == "ethanol" and subsector == "corn ethanol" and technology in ("corn ethanol", "corn ethanol (no constraint)")')

                cornOil = diffsUSA.query('output == "regional biomassOil" and subsector == "corn ethanol" and technology in ("corn ethanol", "corn ethanol (no constraint)")')

                if len(cornOil) > 0:
                    # divide total corn oil by conversion coefficient to compute resulting biodiesel in each timestep
                    cornOilBD = cornOil[yearCols] / BioOilToBiodieselCoef
                    df = df[yearCols].append(cornOilBD)
                    finalFuel = CORN_BIOFUELS
                else:
                    finalFuel = ETOH

            elif shocked == TECH_BIODIESEL:
                df = diffsUSA.query('output == "refining" and subsector == "biomass liquids" and technology == "%s"' % shocked)
                finalFuel = FAME

            else:
                raise PygcamException('Unrecognized fuel shock: %s', shocked)

            self.fuelEJ = fuelEJ = self.sumDiffs(df)

            _logger.debug("   Shock size: {:.2f} EJ {}".format(fuelEJ, self.shocked))

            self.save_carbon_intensity(finalFuel, lucCO2, kyotoCO2e)
            self.save_rebound_metrics(diffsUSA, diffsROW)

            if self.fuelShockFile:
                with open(self.fuelShockFile, 'w') as f:
                    f.write("{:.3f}\n" % fuelEJ)

    def save_carbon_intensity(self, finalFuel, lucCO2, kyotoCO2e):
            ci_luc_only = lucCO2 / self.fuelEJ
            ci_all_ghgs = kyotoCO2e / self.fuelEJ

            ghgResults = {
                'ci-luc-only' : ci_luc_only,
                'ci-all-ghgs' : ci_all_ghgs,
            }

            df = pd.DataFrame(data={'method' : list(ghgResults.keys()), 'value' : list(ghgResults.values())})

            df['Units'] = 'g CO2e/MJ'
            df['baseline'] = self.baseline
            df['policy'] = self.policy

            df.set_index('method', inplace=True)
            df.loc['percent-change', 'Units'] = None    # clear the incorrect units
            df.reset_index(inplace=True)

            _logger.info(log_fmt_df(df))

            ciFile = self.queryPathname('carbon_intensity')
            writeCsv(df, ciFile, header='Carbon intensity', float_format="%.3f")
            df.set_index('method', inplace=True)

            _logger.info(log_fmt_df(df))

    def save_rebound_metrics(self, diffsUSA, diffsROW):
        deltaGlobalEJ = self.sumDiffs(diffsUSA) + self.sumDiffs(diffsROW)

        shock_rebound = deltaGlobalEJ / self.fuelEJ
        _logger.info('%.2f EJ fuel shocked, %.2f EJ change in global fuel use', self.fuelEJ, deltaGlobalEJ)
        _logger.info('Rebound effect: %.1f%%\n', shock_rebound * 100.0)

        def sum_techs(df, techs):
            techsDF = df.query("technology in {}".format(techs))
            total = self.sumDiffs(techsDF)
            return total

        # (delta ROW liquid fossil fuels) / (delta US liquid fossil fuels)
        fossilTechs = ('oil refining', 'coal to liquids', 'gas to liquids')
        deltaFossilROW = sum_techs(diffsROW, fossilTechs)
        deltaFossilUSA = sum_techs(diffsUSA, fossilTechs)
        fossil_liq_ratio = deltaFossilROW / deltaFossilUSA

        # Express these as a percentage, not a fraction
        reboundResults = {
            'fossil-liq-ratio' : 100 * fossil_liq_ratio,
            'shock-rebound'    : 100 * shock_rebound,
        }

        # Save values to CVS to load from results.xml
        reboundFile = self.queryPathname('fuel_rebound')
        reboundDF = pd.DataFrame(data={'name': list(reboundResults.keys()), 'value': list(reboundResults.values())})
        writeCsv(reboundDF, reboundFile, header='Fuel rebound effects', float_format="%.3f", index=None)

        _logger.info(log_fmt_df(reboundDF))

    def save_luc_emissions(self):
        '''
        Save LUC emissions by region, both as total delta and share of global delta.
        '''
        lucCO2 = self.lucCO2

        regions = lucCO2.region.unique()
        luc_by_region = pd.DataFrame(columns=['total', 'share'], index=regions)
        luc_by_region.index.rename('region', inplace=True)

        yearCols = yearColumns(lucCO2)
        landUse2 = lucCO2.set_index('region')

        region_deltas = luc_by_region['total'] = landUse2[yearCols].sum(axis=1)
        global_deltas = region_deltas.sum()

        luc_by_region['share'] = luc_by_region['total'] / global_deltas
        luc_by_region.reset_index(inplace=True)

        pathname = self.queryPathname('luc-CO2-by-region')
        writeCsv(luc_by_region, pathname, header='LUC CO2 emissions by region', float_format="%.3f")

    def save_rfs_categories(self):
        from collections import OrderedDict

        filename = self.queryPathname('RFS-emissions')
        outFile  = filename[:-4] + '-as-CO2e.csv'

        normedDF = self.normedDF
        usa = normedDF.query("region == 'USA'")

        row = normedDF.query("region != 'USA'").groupby('sector').aggregate(np.sum)
        row.reset_index(inplace=True)

        dom_luc = self.lucCO2.query("region == 'USA'")
        row_luc = self.lucCO2.query("region != 'USA'").groupby('sector').aggregate(np.sum)

        def sector_total(df, name):
            sector = df.query("sector == '{}'".format(name))
            total = float(sector.sum(axis=1))
            return total

        farm      = 'Farm Inputs and Fert N2O'
        livestock = 'Livestock'
        fossil    = 'Fossil Energy/Fuel Production'
        other     = 'Other'
        luc       = 'Land Use Change'

        # Map CSV category names to display names
        sectors = {
            'CropProduction|Total' : farm,
            'LivestockProduction' : livestock,
            'EnergyfromFossilFuels|Total' : fossil,
            'OtherIndustrial' : other,
        }

        fuelEJ = self.fuelEJ

        usa_sect = {name : sector_total(usa, sector) for sector, name in sectors.items()}
        row_sect = {name : sector_total(row, sector) for sector, name in sectors.items()}

        usa_sect[luc] = float(dom_luc.sum(axis=1))
        row_sect[luc] = float(row_luc.sum(axis=1))

        total_other = usa_sect[other] + row_sect[other]
        total = sum(usa_sect.values()) + sum(row_sect.values())

        data = OrderedDict()
        include = (farm, luc, livestock, fossil)

        for name in include:
            data['Domestic ' + name] = usa_sect[name]

        for name in include:
            data['International ' + name] = row_sect[name]

        data[other] = total_other
        data['Total'] = total

        df = pd.DataFrame(data={"name": list(data.keys()), "value": list(data.values())})

        u = getUnits()
        df.value = (df.value / fuelEJ) / u.MJ_to_MMBtu

        writeCsv(df, outFile, header='Cumulative GHG emissions (g CO2e/mmBtu)', float_format="%.5f", index=None)

        _logger.info(log_fmt_df(df))

    def save_normalized_GHGs(self):
        '''
        Read a csv file and write a version of it with all currently tracked
        GHGs converted to CO2-equivalents. Add a row for LUC CO2 emissions.
        Return the composed DataFrame.
        '''
        diffs = self.diffDFs

        df = diffs[Q_GHG_EMISSIONS].copy()
        yearCols = yearColumns(df)

        # convert to CO2-equivalents using GWP100 values
        for gas in df.GHG.unique():
            factor = GcamCO2eFactors[gas] if gas in GcamCO2eFactors else 0
            df.loc[df.GHG == gas, yearCols] *= factor

        indexCols = ['region', 'sector']
        grouped = df.groupby(indexCols)
        df = grouped.aggregate(np.sum)
        df.reset_index(inplace=True)

        df = pd.concat([df, self.lucCO2]) # , sort=True)
        df['Units'] = 'Tg CO2e'

        df = df[['region', 'sector'] + yearCols + ['Units']]

        filename = self.queryPathname('GHG-emissions')[:-4] # chop off the '.csv'
        filename += '-as-CO2e.csv'
        writeCsv(df, filename, header='CO2e emissions by aggregated sector', float_format="%.5f")
        return df

def shockedFuel(policy):
    'Guess the fuel type from the policy name.'
    import re

    policy = policy.lower()
    if policy.startswith('corn'):
        return TECH_CORN_ETHANOL

    if re.match(r'^(sugar|cane)', policy):
        return TECH_SUGARCANE_ETHANOL

    if re.match(r'^(cell|biomass|residue|stover|switchgrass|willow|miscan)', policy):
        return TECH_CELLULOSIC_ETHANOL

    if re.match(r'^(biod|soy|canola|jatr)', policy):
        return TECH_BIODIESEL

    if re.match(r'^(FT|ft|fischer)', policy):
        return TECH_FT_BIOFUELS

    # 'grass': 'EnergyGrass cellulosic ethanol',
    # 'palm': 'palm biodiesel',
    # 'wood': 'FT WoodyCrops',

    raise PygcamException("Can't infer shocked fuel fuel from policy name '%s'" % policy)

