class ServiceResult:
    def __init__(self, is_success: bool, data, message: str = None):
        self.is_success: bool = is_success
        self.data = data
        self.message: str = message
