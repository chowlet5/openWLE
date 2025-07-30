from dataclasses import dataclass, fields

class Error(Exception):
    """Base class for exceptions in this module."""
    pass
    
class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

@dataclass
class DataClassInputError:


    def __init__(self, **kwargs):
        # Get the field names of the dataclass
        field_names = {f.name for f in fields(self)}

        # Check for unexpected parameters
        unexpected_params = set(kwargs) - field_names
        if unexpected_params:
            raise TypeError(f"Unexpected parameters: {', '.join(unexpected_params)}. The required parameters are: {', '.join(field_names)}" )

        # Set the attributes
        for field in field_names:
            setattr(self, field, kwargs.get(field))

        

