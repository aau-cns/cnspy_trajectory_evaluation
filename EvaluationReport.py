import configparser


class EvaluationReport:
    directory = None
    fn_gt = None
    fn_est = None
    ANEES_p = None
    ANEES_q = None
    RMSE_p = None
    RMSE_q = None

    def __init__(self, directory='', fn_gt='', fn_est='', ANEES_p=0.0, ANEES_q=0.0, RMSE_p=0.0, RMSE_q=0.0):
        self.directory = directory
        self.fn_gt = fn_gt
        self.fn_est = fn_est
        self.ANEES_p = ANEES_p
        self.ANEES_q = ANEES_q
        self.RMSE_p = RMSE_p
        self.RMSE_q = RMSE_q

    def save(self, fn):
        config = configparser.ConfigParser()
        config['EvaluationReport'] = {'directory': self.directory,
                                      'fn_gt': self.fn_gt,
                                      'fn_est': self.fn_est,
                                      'ANEES_p': self.ANEES_p,
                                      'ANEES_q': self.ANEES_q,
                                      'RMSE_p': self.RMSE_p,
                                      'RMSE_q': self.RMSE_q}
        # print('Save config file....')
        with open(fn, 'w') as configfile:
            config.write(configfile)
            configfile.close()

    def load(self, fn):
        config = configparser.ConfigParser()
        config.sections()
        config.read(fn)
        # print('load from section')
        section = config['EvaluationReport']
        self.directory = section.get('directory', 'default')
        self.fn_gt = section.get('fn_gt', 'default')
        self.fn_est = section.get('fn_est', 'default')
        self.ANEES_p = section.get('ANEES_p', 'default')
        self.ANEES_q = section.get('ANEES_q', 'default')
        self.RMSE_p = section.get('RMSE_p', 'default')
        self.RMSE_q = section.get('RMSE_q', 'default')


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time


class EvaluationReport_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]"

    def get_fn(self):
        fn_gt_csv = "../sample_data/ID1-pose-gt.csv"
        fn_est_csv = "../sample_data/ID1-pose-est-cov.csv"
        return fn_gt_csv, fn_est_csv

    def test_init(self):
        self.start()
        report = EvaluationReport()
        report.directory = ''
        report.fn_gt = "../sample_data/ID1-pose-gt.csv"
        report.fn_est = "../sample_data/ID1-pose-est-cov.csv"
        report.ANEES_p = 0.1
        report.ANEES_q = 0.2
        report.RMSE_p = 0.3
        report.RMSE_q = 0.4

        fn = './eval-report.ini'
        report.save(fn)

        report_ = EvaluationReport()
        report_.load(fn)

        self.assertTrue(report.fn_gt == report_.fn_gt)

        self.stop()


if __name__ == "__main__":
    unittest.main()
