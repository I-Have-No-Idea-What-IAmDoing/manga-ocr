import os
import platform
import pytest
from unittest.mock import patch, MagicMock

# Due to the vendored nature, we might have issues with imports.
# Let's try to import directly and see if it works.
from manga_ocr_dev.vendored.html2image.browsers.search_utils import (
    find_first_defined_env_var,
    find_chrome,
    find_firefox,
    get_command_origin,
    ENV_VAR_LOOKUP_TOGGLE,
    CHROME_EXECUTABLE_ENV_VAR_CANDIDATES,
    FIREFOX_EXECUTABLE_ENV_VAR_CANDIDATES,
)


class TestFindFirstDefinedEnvVar:
    def test_toggle_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            result = find_first_defined_env_var(['VAR1'], ENV_VAR_LOOKUP_TOGGLE)
            assert result is None

    def test_toggle_set_var_exists(self):
        with patch.dict(os.environ, {ENV_VAR_LOOKUP_TOGGLE: '1', 'VAR1': 'value1'}, clear=True):
            result = find_first_defined_env_var(['VAR1'], ENV_VAR_LOOKUP_TOGGLE)
            assert result == 'value1'

    def test_toggle_set_var_does_not_exist(self):
        with patch.dict(os.environ, {ENV_VAR_LOOKUP_TOGGLE: '1'}, clear=True):
            result = find_first_defined_env_var(['VAR1'], ENV_VAR_LOOKUP_TOGGLE)
            assert result is None

    def test_toggle_set_first_of_two_vars_exists(self):
        with patch.dict(os.environ, {ENV_VAR_LOOKUP_TOGGLE: '1', 'VAR1': 'value1', 'VAR2': 'value2'}, clear=True):
            result = find_first_defined_env_var(['VAR1', 'VAR2'], ENV_VAR_LOOKUP_TOGGLE)
            assert result == 'value1'

    def test_toggle_set_second_of_two_vars_exists(self):
        with patch.dict(os.environ, {ENV_VAR_LOOKUP_TOGGLE: '1', 'VAR2': 'value2'}, clear=True):
            result = find_first_defined_env_var(['VAR1', 'VAR2'], ENV_VAR_LOOKUP_TOGGLE)
            assert result == 'value2'


class TestFindChrome:

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var')
    def test_find_from_env_var(self, mock_find_env):
        mock_find_env.return_value = '/path/from/env/chrome'
        result = find_chrome()
        assert result == '/path/from/env/chrome'
        mock_find_env.assert_called_once_with(
            env_var_list=CHROME_EXECUTABLE_ENV_VAR_CANDIDATES,
            toggle=ENV_VAR_LOOKUP_TOGGLE
        )

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('shutil.which')
    def test_find_on_linux_which(self, mock_which, mock_system, mock_find_env):
        mock_which.side_effect = lambda cmd: '/usr/bin/chromium' if cmd == 'chromium' else None
        result = find_chrome()
        assert result == 'chromium'
        mock_which.assert_called_once_with('chromium')

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('shutil.which', return_value=None)
    @patch('subprocess.check_output', side_effect=FileNotFoundError)
    def test_not_found_linux(self, mock_check_output, mock_which, mock_system, mock_find_env):
        with pytest.raises(FileNotFoundError):
            find_chrome()

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Darwin')
    @patch('subprocess.check_output')
    def test_find_on_macos(self, mock_check_output, mock_system, mock_find_env):
        mock_check_output.return_value = b'Google Chrome 1.2.3'
        result = find_chrome()
        assert result == '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Windows')
    @patch('os.getenv')
    @patch('os.path.isfile')
    def test_find_on_windows(self, mock_isfile, mock_getenv, mock_system, mock_find_env):
        program_files = 'C:\\Program Files'
        mock_getenv.side_effect = ['C:\\Program Files (x86)', program_files, 'C:\\Users\\User\\AppData\\Local']
        mock_isfile.side_effect = [False, True, False]
        result = find_chrome()
        expected = os.path.join(program_files, 'Google\\Chrome\\Application\\chrome.exe')
        assert result == expected

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('subprocess.check_output')
    def test_with_user_given_executable_valid(self, mock_check_output, mock_system, mock_find_env):
        mock_check_output.return_value = b'Chromium 1.2.3'
        result = find_chrome(user_given_executable='/usr/bin/my_chrome')
        assert result == '/usr/bin/my_chrome'
        mock_check_output.assert_called_with(['/usr/bin/my_chrome', '--version'])

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('subprocess.check_output', side_effect=Exception)
    def test_with_user_given_executable_invalid(self, mock_check_output, mock_system, mock_find_env):
        with pytest.raises(FileNotFoundError):
            find_chrome(user_given_executable='/usr/bin/not_chrome')


class TestFindFirefox:

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var')
    def test_find_from_env_var(self, mock_find_env):
        mock_find_env.return_value = '/path/from/env/firefox'
        result = find_firefox()
        assert result == '/path/from/env/firefox'
        mock_find_env.assert_called_once_with(
            env_var_list=FIREFOX_EXECUTABLE_ENV_VAR_CANDIDATES,
            toggle=ENV_VAR_LOOKUP_TOGGLE
        )

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('shutil.which')
    def test_find_on_linux_which(self, mock_which, mock_system, mock_find_env):
        mock_which.return_value = '/usr/bin/firefox'
        result = find_firefox()
        assert result == 'firefox'
        mock_which.assert_called_with('firefox')

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('shutil.which', return_value=None)
    def test_not_found_linux(self, mock_which, mock_system, mock_find_env):
        with pytest.raises(FileNotFoundError):
            find_firefox()

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Darwin')
    @patch('subprocess.check_output')
    def test_find_on_macos(self, mock_check_output, mock_system, mock_find_env):
        mock_check_output.return_value = b'Mozilla Firefox 1.2.3'
        result = find_firefox()
        assert result == '/Applications/Firefox.app/Contents/MacOS/firefox'

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Windows')
    @patch('os.getenv')
    @patch('os.path.isfile')
    def test_find_on_windows(self, mock_isfile, mock_getenv, mock_system, mock_find_env):
        program_files = 'C:\\Program Files'
        mock_getenv.side_effect = ['C:\\Program Files (x86)', program_files, 'C:\\Users\\User\\AppData\\Local']
        mock_isfile.side_effect = [False, True, False]
        result = find_firefox()
        expected = os.path.join(program_files, 'Mozilla Firefox\\firefox.exe')
        assert result == expected

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('subprocess.check_output')
    def test_with_user_given_executable_valid(self, mock_check_output, mock_system, mock_find_env):
        mock_check_output.return_value = b'Mozilla Firefox 1.2.3'
        result = find_firefox(user_given_executable='/usr/bin/my_firefox')
        assert result == '/usr/bin/my_firefox'
        mock_check_output.assert_called_with(['/usr/bin/my_firefox', '--version'])

    @patch('manga_ocr_dev.vendored.html2image.browsers.search_utils.find_first_defined_env_var', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('subprocess.check_output', side_effect=Exception)
    def test_with_user_given_executable_invalid(self, mock_check_output, mock_system, mock_find_env):
        with pytest.raises(FileNotFoundError):
            find_firefox(user_given_executable='/usr/bin/not_firefox')